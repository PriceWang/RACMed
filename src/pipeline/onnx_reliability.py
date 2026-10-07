"""ONNX Runtime reliability evaluation for segmentation models."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.datasets.dataset_factory import create_dataloaders
from src.metrics.segmentation_metrics import segmentation_metrics
from src.metrics.classification_metrics import classification_metrics
from src.models.forward import forward_model
from src.models.segmentation2d import rasterize_box_prompts
from src.utils.progress import progress_bar


def deployment_segmentation_threshold(
    config: dict[str, Any], artifact_root: str | Path | None = None
) -> float | list[float] | None:
    """Return a validation-frozen threshold for fair backend comparisons."""

    value=config.get("evaluation",{}).get("fixed_segmentation_threshold")
    if value is None and artifact_root is not None:
        path=Path(artifact_root)/"segmentation_threshold_calibration.json"
        if path.exists():value=json.loads(path.read_text()).get("threshold")
    if value is None:return None
    return [float(item) for item in value] if isinstance(value,(list,tuple)) else float(value)


def mean_segmentation_case_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average per-case numeric metrics using the PyTorch evaluator convention."""

    if not rows:
        raise ValueError("ONNX reliability evaluation received zero cases.")
    return {
        key:float(np.mean([float(row[key]) for row in rows]))
        for key,value in rows[0].items()
        if isinstance(value,(int,float))
    }


def reliability_score(metrics: dict[str, Any]) -> float:
    """Composite segmentation reliability score used in RACMed reports."""
    return float(
        metrics["dice"]
        + 0.05 * metrics["boundary_f1"]
        + 0.10 * metrics["small_component_recall"]
        - 0.10 * metrics["false_negative_lesion_rate"]
        - 0.001 * metrics.get("assd",0.0)
    )


def deployment_classification_calibration(
    config:dict[str,Any],artifact_root:str|Path|None=None
) -> dict[str,float]:
    """Load validation-frozen Platt scaling and decision threshold."""

    value=config.get("evaluation",{}).get("fixed_classification_calibration")
    if value is None and artifact_root is not None:
        path=Path(artifact_root)/"classification_calibration.json"
        if path.exists():value=json.loads(path.read_text())
    value=value or {}
    return {"scale":float(value.get("scale",1.0)),"bias":float(value.get("bias",0.0)),"threshold":float(value.get("threshold",0.5))}


def evaluate_onnx_classification(
    config:dict[str,Any],onnx_path:str|Path,debug:bool=False,progress_label:str|None=None
) -> dict[str,Any]:
    """Evaluate classification ONNX artifacts with frozen validation calibration."""

    import onnxruntime as ort

    session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
    input_name=session.get_inputs()[0].name
    loader=create_dataloaders(config,debug)["test"]
    calibration=deployment_classification_calibration(config,config.get("output_dir"))
    logits=[];targets=[];times=[]
    batches=progress_bar(loader,desc=f"ONNX eval {progress_label or 'model'}",total=len(loader),unit="batch",leave=True)
    for batch in batches:
        image=batch["image"].numpy().astype("float32");start=time.perf_counter()
        output=session.run(None,{input_name:image})[0]
        times.append((time.perf_counter()-start)*1000.)
        logits.append(torch.from_numpy(output)*calibration["scale"]+calibration["bias"])
        targets.append(batch["target"])
    if not logits:raise ValueError("ONNX classification evaluation received zero batches.")
    metrics=classification_metrics(torch.cat(logits),torch.cat(targets),threshold=calibration["threshold"])
    values=np.asarray(times,dtype=np.float64)
    metrics.update({"latency_mean_ms_per_batch":float(values.mean()),"latency_p50_ms_per_batch":float(np.percentile(values,50)),"latency_p95_ms_per_batch":float(np.percentile(values,95)),"provider":session.get_providers()[0],"calibration":calibration})
    return metrics


def _onnx_segmentation_forward(
    session:Any,
    session_inputs:list[Any],
    image:torch.Tensor,
    prompt_maps:torch.Tensor|None=None,
    overlap:float=0.25,
    sw_batch_size:int=1,
) -> torch.Tensor:
    """Run a fixed-shape ONNX crop over an arbitrary 2D/3D image.

    Training models may accept a full volume while their deployment artifact is
    intentionally exported for a bounded crop. When spatial dimensions differ,
    MONAI sliding-window blending reconstructs the full output using the ONNX
    graph's declared spatial input size. This keeps deployment evaluation inside
    the real artifact contract instead of feeding an invalid full-volume shape.
    """

    input_name=session_inputs[0].name
    declared=tuple(session_inputs[0].shape[2:])
    fixed_spatial=tuple(int(value) for value in declared) if declared and all(isinstance(value,int) and value>0 for value in declared) else None
    needs_sliding=fixed_spatial is not None and tuple(image.shape[2:])!=fixed_spatial
    if needs_sliding:
        if len(session_inputs)>1 or prompt_maps is not None:
            raise ValueError("Sliding-window ONNX inference with prompt-map inputs is unsupported.")
        from monai.inferers import sliding_window_inference

        declared_batch=session_inputs[0].shape[0]
        runtime_batch=int(declared_batch) if isinstance(declared_batch,int) and declared_batch>0 else max(int(sw_batch_size),1)

        def predictor(window:torch.Tensor) -> torch.Tensor:
            values=window.detach().cpu().numpy().astype("float32",copy=False)
            return torch.from_numpy(session.run(None,{input_name:values})[0])

        with torch.inference_mode():
            return sliding_window_inference(
                image.float(),roi_size=fixed_spatial,sw_batch_size=runtime_batch,
                predictor=predictor,overlap=float(overlap),
            )
    feed={input_name:image.numpy().astype("float32")}
    if prompt_maps is not None:feed[session_inputs[1].name]=prompt_maps.numpy().astype("float32")
    return torch.from_numpy(session.run(None,feed)[0])


def evaluate_onnx_segmentation(
    config: dict[str, Any],
    onnx_path: str | Path,
    debug: bool = False,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """Evaluate an ONNX segmentation model with RACMed reliability metrics."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised when optional dep missing
        raise ImportError("onnxruntime is required for ONNX reliability evaluation") from exc

    requested_provider=str(config.get("deployment",{}).get("reliability_provider","auto")).lower()
    available=set(ort.get_available_providers())
    if requested_provider not in {"auto","cpu","cuda"}:
        raise ValueError("deployment.reliability_provider must be auto, cpu, or cuda")
    use_cuda=(
        requested_provider=="cuda"
        or (requested_provider=="auto" and torch.cuda.is_available())
    ) and "CUDAExecutionProvider" in available
    if requested_provider=="cuda" and not use_cuda:
        raise RuntimeError(
            "CUDA ONNX reliability evaluation was requested, but CUDAExecutionProvider is unavailable."
        )
    providers=["CUDAExecutionProvider","CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    if progress_label:
        from src.pipeline.progress import pipeline_logger
        pipeline_logger().info(
            "ONNX_EVAL_SESSION | backend=%s providers=%s surface_metrics=%s",
            progress_label,",".join(session.get_providers()),
            bool(config.get("deployment",{}).get("reliability_surface_metrics",False)),
        )
    session_inputs = session.get_inputs()
    input_name = session_inputs[0].name
    case_metrics: list[dict[str, Any]] = []
    times: list[float] = []
    num_classes=int(config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 2)))
    threshold=deployment_segmentation_threshold(config,config.get("output_dir"))
    small_component_threshold=int(config.get("evaluation",{}).get("small_component_threshold",config.get("risk",{}).get("small_component_threshold",64)))
    include_surface=bool(config.get("deployment",{}).get("reliability_surface_metrics",False))
    loader=create_dataloaders(config, debug)["test"]
    sliding=config.get("evaluation",{}).get("sliding_window",{}) or {}
    validation_cases=config.get("deployment",{}).get("reliability_validation_cases")
    max_cases=int(validation_cases) if validation_cases is not None else None
    if max_cases is not None and max_cases<1:
        raise ValueError("deployment.reliability_validation_cases must be a positive integer.")
    current_group=None;group_outputs=[];group_targets=[]
    stopped_at_case_limit=False

    def flush_group() -> None:
        nonlocal current_group,group_outputs,group_targets
        if not group_outputs:return
        if group_outputs[0].ndim==3:
            output=torch.stack(group_outputs,dim=1).unsqueeze(0)
            target_stack_dim=1 if group_targets[0].ndim==group_outputs[0].ndim and group_targets[0].shape[0]==group_outputs[0].shape[0] else 0
            target=torch.stack(group_targets,dim=target_stack_dim).unsqueeze(0)
        else:
            output=group_outputs[0].unsqueeze(0);target=group_targets[0].unsqueeze(0)
        case_metrics.append(segmentation_metrics(
            output,target,num_classes,threshold=threshold,
            small_component_threshold=small_component_threshold,
            include_surface=include_surface,
        ))
        group_outputs=[];group_targets=[]
    total_batches=len(loader);progress_interval=max(1,total_batches//10) if total_batches else 1
    batches=progress_bar(loader,desc=f"ONNX eval {progress_label or 'model'}",total=total_batches,unit="batch",leave=True)
    for step,batch in enumerate(batches,start=1):
        group_ids=batch.get("group_id",batch["id"])
        first_group=str(group_ids[0])
        if current_group is not None and first_group!=current_group:
            flush_group()
            if max_cases is not None and len(case_metrics)>=max_cases:
                stopped_at_case_limit=True
                break
        start = time.perf_counter()
        prompt_maps=None
        if len(session_inputs) > 1:
            if "boxes" not in batch:
                raise ValueError("Prompt-aware ONNX model requires boxes in the evaluation batch.")
            prompt_channels=int(config.get("compressed_model",{}).get("prompt_channels",max(int(config.get("model",{}).get("num_classes",2))-1,1)))
            prompt_maps=rasterize_box_prompts(batch["image"],batch["boxes"],batch.get("box_valid"),prompt_channels)
        output=_onnx_segmentation_forward(
            session,session_inputs,batch["image"],prompt_maps,
            overlap=float(sliding.get("overlap",0.25)),
            sw_batch_size=int(sliding.get("sw_batch_size",1)),
        )
        times.append((time.perf_counter() - start) * 1000.0)
        target=batch["target"]
        for index in range(output.shape[0]):
            group_id=str(group_ids[index])
            if current_group is not None and group_id!=current_group:
                flush_group()
                if max_cases is not None and len(case_metrics)>=max_cases:
                    stopped_at_case_limit=True
                    break
            current_group=group_id;group_outputs.append(output[index]);group_targets.append(target[index])
        if stopped_at_case_limit:break
        if progress_label and (step%progress_interval==0 or step==total_batches):
            from src.pipeline.progress import pipeline_logger
            pipeline_logger().info("ONNX_EVAL_PROGRESS | backend=%s batch=%d/%d",progress_label,step,total_batches)
    if not stopped_at_case_limit:flush_group()
    metrics = mean_segmentation_case_metrics(case_metrics)
    metrics["evaluated_cases"]=len(case_metrics)
    metrics["evaluation_scope"]="deployment_validation_subset" if max_cases is not None else "full_test"
    metrics["reliability_score"] = reliability_score(metrics)
    values = np.asarray(times, dtype=np.float64)
    metrics["latency_mean_ms_per_batch"] = float(values.mean())
    metrics["latency_p50_ms_per_batch"] = float(np.percentile(values, 50))
    metrics["latency_p95_ms_per_batch"] = float(np.percentile(values, 95))
    metrics["provider"] = session.get_providers()[0]
    metrics["providers"] = session.get_providers()
    metrics["surface_metrics_included"] = include_surface
    return metrics


def save_onnx_reliability(config: dict[str, Any], paths: dict[str, str | Path], output_path: str | Path, debug: bool = False) -> dict[str, Any]:
    """Evaluate and save reliability metrics for one or more ONNX artifacts."""
    results: dict[str, Any] = {}
    for label, path in paths.items():
        from src.pipeline.progress import pipeline_logger
        pipeline_logger().info("ONNX_EVAL_START | backend=%s",label)
        try:
            if config.get("task_type")=="classification":
                results[label]=evaluate_onnx_classification(config,path,debug,label)
                pipeline_logger().info("ONNX_EVAL_DONE | backend=%s auc=%.4f ece=%.4f",label,results[label]["auc"],results[label]["ece"])
            else:
                results[label] = evaluate_onnx_segmentation(config, path, debug, label)
                pipeline_logger().info(
                    "ONNX_EVAL_DONE | backend=%s dice=%.4f small_recall=%.4f fn_lesion_rate=%.4f",
                    label,
                    results[label]["dice"],
                    results[label]["small_component_recall"],
                    results[label]["false_negative_lesion_rate"],
                )
        except Exception as exc:  # pragma: no cover - user-facing fallback
            results[label] = {"error": type(exc).__name__, "message": str(exc)}
            pipeline_logger().warning("ONNX_EVAL_FAIL | backend=%s error=%s",label,exc)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    rows = []
    for label, metrics in results.items():
        row = {"backend": label}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    result_root = output.parent / "results"
    if result_root.exists():
        shutil.rmtree(result_root)
    for label, metrics in results.items():
        if "error" in metrics:
            continue
        directory = result_root / label
        directory.mkdir(parents=True, exist_ok=True)
        row = {
            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
            "method": label,
            "mode": "deployment",
            "task": config.get("task_type", "segmentation2d"),
            "seed": config.get("seed", 0),
            "score": metrics.get("auc",metrics.get("dice", 0.0)),
            "latency_mean_ms": metrics.get("latency_mean_ms_per_batch"),
            "latency_p50_ms": metrics.get("latency_p50_ms_per_batch"),
            "latency_p95_ms": metrics.get("latency_p95_ms_per_batch"),
        }
        pd.DataFrame([row]).to_csv(directory / "summary.csv", index=False)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return results
