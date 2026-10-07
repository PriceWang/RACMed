"""ONNX, quantization, and CoreML export stage."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from src.compression.elastic_policy import ElasticCompressionPolicy
from src.compression.onnx_export import export_coreml, export_onnx, export_tensorrt_hook
from src.compression.quantization import InMemoryCalibrationReader,onnx_int8,onnx_qdq_int8,onnx_selective_int8,onnx_selective_qdq_int8
from src.engine.checkpoint import load_model_checkpoint
from src.models.model_factory import create_model
from src.pipeline.model_artifacts import example_input_shape, export_config, pipeline_checkpoint, pipeline_model_role
from src.pipeline.progress import pipeline_logger
from src.utils.progress import progress_bar


def _example(model: torch.nn.Module, shape: list[int]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(*shape)
    if getattr(model, "accepts_prompts", False):
        channels = int(getattr(model, "prompt_channels", 1))
        return image, torch.zeros(shape[0], channels, *shape[2:])
    return image


def _qdq_calibration_reader(config:dict[str,Any],model_cfg:dict[str,Any],model:torch.nn.Module,debug:bool) -> InMemoryCalibrationReader:
    """Collect deterministic non-test batches for TensorRT Q/DQ calibration."""

    from src.datasets.dataset_factory import create_dataloaders

    deployment=config.get("deployment",{})
    split=str(deployment.get("qdq_calibration_split","train"))
    if split not in {"train","val"}:raise ValueError("Q/DQ calibration split must be train or val; test is forbidden.")
    maximum=int(deployment.get("qdq_calibration_batches",8 if not debug else 2))
    loaders=create_dataloaders(model_cfg,debug)
    loader=loaders[split]
    records=[]
    total=min(len(loader),maximum) if hasattr(loader,"__len__") else maximum
    role=pipeline_model_role(model_cfg,"deployment")
    expected_spatial=tuple(example_input_shape(model_cfg,role)[2:])
    for batch in progress_bar(islice(loader,maximum),desc="TensorRT Q/DQ calibration",total=total,unit="batch",leave=True):
        images=batch["image"][:1].detach().cpu().float()
        prompts=None
        if getattr(model,"accepts_prompts",False):
            if "prompt_maps" in batch:
                prompts=batch["prompt_maps"][:1].detach().cpu().float()
            elif "boxes" in batch and hasattr(model,"_prompt_maps"):
                boxes=batch["boxes"][:1].detach().cpu()
                valid=batch.get("box_valid")
                valid=valid[:1].detach().cpu() if valid is not None else None
                prompts=model._prompt_maps(images,boxes,valid).float()
            else:
                raise ValueError("Prompt-aware Q/DQ calibration requires prompt_maps or boxes in each batch.")
        if tuple(images.shape[2:])!=expected_spatial:
            mode="bilinear" if images.ndim==4 else "trilinear"
            images=F.interpolate(images,size=expected_spatial,mode=mode,align_corners=False)
            if prompts is not None:prompts=F.interpolate(prompts,size=expected_spatial,mode="nearest")
        record={"input":images.numpy()}
        if prompts is not None:
            record["prompt_maps"]=prompts.numpy()
        records.append(record)
    pipeline_logger().info("QDQ_CALIBRATION | split=%s batches=%d",split,len(records))
    return InMemoryCalibrationReader(records)


def _layer_scores(config: dict[str, Any], model_cfg: dict[str, Any], model: torch.nn.Module, role: str, path: Path, debug: bool) -> dict[str, Any]:
    if path.exists():
        return {row.layer: row._asdict() for row in pd.read_csv(path).itertuples(index=False)}
    # Risk profiling and dataset construction are training-side dependencies.
    # Import them only when a deployment bundle does not already contain its
    # persisted layer-risk scores.
    from src.compression.layer_risk import LayerRiskProfiler
    from src.datasets.dataset_factory import create_dataloaders

    split = config.get("compression", {}).get("profile_split") or ("train" if role != "compressed_model" else "val")
    batch = next(iter(create_dataloaders(model_cfg, debug)[split]))
    forward_kwargs = {key: batch[key] for key in ("boxes", "box_valid") if key in batch} or None
    scores = LayerRiskProfiler().profile(
        model,
        batch["image"],
        max_layers=config.get("compression", {}).get("max_profile_layers"),
        forward_kwargs=forward_kwargs,
    )
    LayerRiskProfiler.save(scores, path)
    return scores


def run_export(
    config: dict[str, Any],
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
    dynamic_axes: bool = False,
    evaluate_onnx: bool = False,
    coreml: bool = False,
    tensorrt: bool = False,
    debug: bool = False,
    tensorrt_qdq: bool = False,
    coreml_only: bool = False,
) -> dict[str, str]:
    """Export deployable graphs and execute uniform/risk-aware quantization."""

    cfg = dict(config)
    cfg["evaluation"] = dict(config.get("evaluation", {}))
    out = Path(cfg.get("output_dir", "outputs/experiment"))
    comp, deploy = out / "compression", out / "deployment"
    comp.mkdir(parents=True, exist_ok=True)
    deploy.mkdir(parents=True, exist_ok=True)
    role = pipeline_model_role(cfg, "deployment")
    model_cfg = export_config(cfg, role)
    model = create_model(model_cfg, role)
    selected = pipeline_checkpoint(cfg, out, "deployment", checkpoint)
    if selected is not None:
        if not selected.exists():
            raise FileNotFoundError(f"Deployment checkpoint not found: {selected}")
        model,_=load_model_checkpoint(selected,model)
    if cfg.get("task_type","").startswith("segmentation") and cfg["evaluation"].get("fixed_segmentation_threshold") is None:
        threshold_paths=[out/"segmentation_threshold_calibration.json"]
        if selected is not None:threshold_paths.append(selected.parent.parent/"segmentation_threshold_calibration.json")
        for threshold_path in threshold_paths:
            if threshold_path.exists():
                cfg["evaluation"]["fixed_segmentation_threshold"]=json.loads(threshold_path.read_text()).get("threshold")
                pipeline_logger().info("EXPORT | frozen_validation_threshold=%s source=%s",cfg["evaluation"]["fixed_segmentation_threshold"],threshold_path)
                break
    if cfg.get("task_type")=="classification" and cfg["evaluation"].get("fixed_classification_calibration") is None:
        calibration_paths=[out/"classification_calibration.json"]
        if selected is not None:calibration_paths.append(selected.parent.parent/"classification_calibration.json")
        for calibration_path in calibration_paths:
            if calibration_path.exists():
                cfg["evaluation"]["fixed_classification_calibration"]=json.loads(calibration_path.read_text())
                pipeline_logger().info("EXPORT | frozen_classification_calibration=%s",calibration_path)
                break
    shape = example_input_shape(model_cfg, role)
    example = _example(model, shape)
    if coreml_only:
        if not coreml:
            raise ValueError("coreml_only requires coreml=True")
        coreml_path=deploy/"model.mlpackage"
        export_coreml(model,example,coreml_path)
        manifest_path=deploy/"artifacts.json"
        try:manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        except json.JSONDecodeError:manifest={}
        manifest["coreml"]=str(coreml_path)
        manifest_path.write_text(json.dumps(manifest,indent=2))
        pipeline_logger().info("EXPORT | format=coreml_only artifact=%s",coreml_path.name)
        pipeline_logger().info("EXPORT_RESULT | artifacts=coreml")
        return {"coreml":str(coreml_path)}
    onnx_path = Path(output) if output else deploy / "model.onnx"
    export_onnx(model, example, onnx_path, dynamic_axes or bool(cfg.get("deployment", {}).get("dynamic_axes", False)))
    pipeline_logger().info("EXPORT | format=onnx_fp32 artifact=%s", onnx_path.name)
    score_path = comp / "layer_risk_scores.csv"
    scores = _layer_scores(cfg, model_cfg, model, role, score_path, debug)
    _, precision = ElasticCompressionPolicy(
        high_threshold=float(cfg.get("compression", {}).get("high_threshold", 0.7)),
        protected_layer_patterns=tuple(cfg.get("compression", {}).get("protected_layer_patterns", ("head", "classifier", ".out."))),
        max_int8_fraction=cfg.get("compression", {}).get("max_int8_fraction"),
    ).build(scores)
    (comp / "mixed_precision_policy.json").write_text(json.dumps(precision, indent=2))
    int8_path, selective_path = deploy / "model.uniform.int8.onnx", deploy / "model.risk_aware.int8.onnx"
    try:
        onnx_int8(onnx_path, int8_path)
        metadata = onnx_selective_int8(
            onnx_path,
            selective_path,
            int8_layers=[name for name, value in precision.items() if value == "int8"],
        )
        (deploy / "risk_aware_quantization_metadata.json").write_text(json.dumps(metadata, indent=2))
        pipeline_logger().info(
            "QUANTIZATION | uniform_int8=true risk_aware_int8=true selected_nodes=%d quantized_nodes=%d",
            metadata["selected_node_count"],
            metadata["quantized_node_count"],
        )
    except Exception as exc:
        (deploy / "onnx_int8_warning.json").write_text(json.dumps({"warning": str(exc)}, indent=2))
        pipeline_logger().warning("QUANTIZATION | ONNX INT8 fallback warning=%s", exc)
    artifacts = {"onnx_fp32": str(onnx_path)}
    if int8_path.exists():
        artifacts["onnx_uniform_int8"] = str(int8_path)
    if selective_path.exists():
        artifacts["onnx_risk_aware_int8"] = str(selective_path)
    if tensorrt_qdq or bool(cfg.get("deployment",{}).get("export_tensorrt_qdq",False)):
        reader=_qdq_calibration_reader(cfg,model_cfg,model,debug)
        uniform_qdq=deploy/"model.uniform.qdq.int8.onnx"
        risk_qdq=deploy/"model.risk_aware.qdq.int8.onnx"
        onnx_qdq_int8(onnx_path,uniform_qdq,reader)
        qdq_metadata=onnx_selective_qdq_int8(
            onnx_path,risk_qdq,
            [name for name,value in precision.items() if value=="int8"],reader,
        )
        (deploy/"risk_aware_qdq_quantization_metadata.json").write_text(json.dumps(qdq_metadata,indent=2))
        artifacts.update({"onnx_uniform_qdq_int8":str(uniform_qdq),"onnx_risk_aware_qdq_int8":str(risk_qdq)})
        pipeline_logger().info(
            "QUANTIZATION | tensorrt_qdq=true selected_nodes=%d q_nodes=%d",
            qdq_metadata["selected_node_count"],qdq_metadata["quantize_linear_nodes"],
        )
    if coreml:
        coreml_path = deploy / "model.mlpackage"
        export_coreml(model, example, coreml_path)
        artifacts["coreml"] = str(coreml_path)
        pipeline_logger().info("EXPORT | format=coreml artifact=%s", coreml_path.name)
    if tensorrt:
        engine = export_tensorrt_hook(onnx_path, deploy / "model.engine")
        if engine is not None:
            artifacts["tensorrt"] = str(engine)
            pipeline_logger().info("EXPORT | format=tensorrt artifact=%s", engine.name)
    if evaluate_onnx:
        # Full reliability evaluation needs dataset and metric dependencies;
        # plain artifact export on an edge device does not.
        from src.pipeline.onnx_reliability import save_onnx_reliability

        onnx_artifacts={key:Path(value) for key,value in artifacts.items() if key.startswith("onnx_")}
        requested=cfg.get("deployment",{}).get("reliability_artifacts")
        if requested is not None:
            requested=[str(label) for label in requested]
            missing=sorted(set(requested)-set(onnx_artifacts))
            if missing:
                raise ValueError(
                    "Requested ONNX reliability artifacts were not exported: "
                    + ", ".join(missing)
                )
            onnx_artifacts={label:onnx_artifacts[label] for label in requested}
        pipeline_logger().info(
            "ONNX_EVAL | backends=%d selected=%s",
            len(onnx_artifacts),
            ",".join(onnx_artifacts),
        )
        save_onnx_reliability(
            cfg,
            onnx_artifacts,
            deploy / "onnx_reliability_metrics.json",
            debug,
        )
    (deploy / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    pipeline_logger().info("EXPORT_RESULT | artifacts=%s", ",".join(sorted(artifacts)))
    return artifacts
