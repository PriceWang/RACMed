"""Resumable orchestration for the validated RACMed experiment matrix."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Iterable

from src.engine.config import load_config, merged
from src.pipeline.analysis import analyze_results
from src.pipeline.benchmarking import run_benchmark,run_onnx_benchmark_suite
from src.pipeline.exporting import run_export
from src.pipeline.evaluation import evaluate_checkpoint
from src.pipeline.pretrained_teacher import evaluate_pretrained_teacher, teacher_output_dir
from src.pipeline.progress import log_skip, logged_stage, pipeline_logger
from src.pipeline.pruning import run_pruning
from src.pipeline.training import train_method
from src.pipeline.layout import ablation_run,compression_run,operating_point_run,training_run
from src.utils.progress import progress_bar


def _summary_has_current_reliability_schema(path:Path,task_type:str) -> bool:
    """Return whether a result contains the calibrated segmentation metrics."""

    if not task_type.startswith("segmentation"):return path.exists()
    if not path.exists():return False
    try:
        import pandas as pd
        columns=set(pd.read_csv(path,nrows=1).columns)
    except Exception:
        return False
    return {"ece","adaptive_ece","brier","nll","small_component_threshold"}.issubset(columns)


def _source_validation_operating_point(
    run: dict[str, Any], source: Path | None
) -> dict[str, Any] | None:
    """Freeze a source model's validation-derived operating point.

    Compression comparisons should not gain or lose performance merely because
    the operating point was re-tuned after pruning.  For classification this
    includes Platt scaling and the decision threshold; for segmentation it is
    the class threshold.  The policy uses only validation artifacts and never
    inspects test predictions.
    """

    evaluation=run.setdefault("evaluation",{})
    task_type=str(run.get("task_type",""))
    policy_key=(
        "compression_calibration_policy"
        if task_type=="classification"
        else "compression_threshold_policy"
    )
    policy=str(evaluation.get(policy_key,"recalibrate"))
    if policy != "source_validation" or source is None:
        return None
    artifact_name=(
        "classification_calibration.json"
        if task_type=="classification"
        else "segmentation_threshold_calibration.json"
    )
    calibration=source.parent.parent/artifact_name
    if not calibration.exists():
        pipeline_logger().warning(
            "COMPRESSION_OPERATING_POINT | source validation threshold unavailable | source=%s",
            source.parent.parent,
        )
        return None
    calibration_payload=json.loads(calibration.read_text())
    threshold=calibration_payload.get("threshold")
    if threshold is None:
        pipeline_logger().warning(
            "COMPRESSION_OPERATING_POINT | threshold missing | artifact=%s",calibration
        )
        return None
    if task_type=="classification":
        evaluation["classification_logit_calibration"]=False
        evaluation["fixed_classification_calibration"]={
            key:calibration_payload[key] for key in ("scale","bias","threshold")
        }
    else:
        evaluation["segmentation_threshold_calibration"]=False
        evaluation["fixed_segmentation_threshold"]=threshold
    payload={"policy":policy,"threshold":threshold,"source_artifact":str(calibration)}
    pipeline_logger().info(
        "COMPRESSION_OPERATING_POINT | policy=%s threshold=%s source=%s",
        policy,
        threshold,
        calibration,
    )
    return payload


def _marker_matches(path: Path, expected: dict[str, Any] | None) -> bool:
    """Return whether a persisted operating-point marker matches the run."""

    if expected is None:
        return True
    if not path.exists():
        return False
    try:
        observed=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return False
    return observed.get("policy")==expected.get("policy") and observed.get("threshold")==expected.get("threshold")


def _benchmark_marker_matches(path: Path, artifact_map: dict[str, str]) -> bool:
    """Invalidate a benchmark marker when the exported artifact set changes."""

    if not path.exists():
        return False
    try:
        observed=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return False
    onnx_variants=sorted(label for label in artifact_map if label.startswith("onnx_"))
    observed_variants=sorted(
        label for label in observed.get("onnx_variants",[]) if str(label).startswith("onnx_")
    )
    return observed_variants==onnx_variants


def _artifact_manifest(deploy:Path) -> dict[str,str]:
    """Read an export manifest only when every declared artifact still exists."""

    path=deploy/"artifacts.json"
    if not path.exists():return {}
    try:
        artifacts=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return {}
    if not isinstance(artifacts,dict):return {}
    return {
        str(label):str(artifact)
        for label,artifact in artifacts.items()
        if Path(str(artifact)).exists()
    }


def _onnx_reliability_complete(
    config:dict[str,Any],deploy:Path,artifacts:dict[str,str]
) -> bool:
    """Reject partial, failed, or validation-subset deployment evaluations."""

    path=deploy/"onnx_reliability_metrics.json"
    if not path.exists():return False
    try:
        metrics=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return False
    requested=config.get("deployment",{}).get("reliability_artifacts")
    desired=(
        [str(label) for label in requested]
        if requested is not None
        else [label for label in artifacts if label.startswith("onnx_")]
    )
    if not desired:return False
    if not set(desired).issubset(artifacts):return False
    for label in desired:
        result=metrics.get(label)
        if not isinstance(result,dict) or "error" in result:return False
        if str(config.get("task_type","")).startswith("segmentation"):
            limit=config.get("deployment",{}).get("reliability_validation_cases")
            expected_scope="deployment_validation_subset" if limit is not None else "full_test"
            if result.get("evaluation_scope")!=expected_scope:return False
            if not isinstance(result.get("evaluated_cases"),int):return False
            if limit is not None and result["evaluated_cases"]!=int(limit):return False
            expected_surface=bool(config.get("deployment",{}).get("reliability_surface_metrics",False))
            if result.get("surface_metrics_included") is not expected_surface:return False
    return True


def _export_complete(
    config:dict[str,Any],
    deploy:Path,
    evaluate_onnx:bool,
    export_qdq:bool,
    operating_point:dict[str,Any]|None=None,
) -> bool:
    """Return whether the exact requested export contract is persisted."""

    artifacts=_artifact_manifest(deploy)
    if "onnx_fp32" not in artifacts:return False
    if export_qdq and not {
        "onnx_uniform_qdq_int8","onnx_risk_aware_qdq_int8"
    }.issubset(artifacts):
        return False
    if evaluate_onnx and not _onnx_reliability_complete(config,deploy,artifacts):
        return False
    return _marker_matches(deploy/"compression_operating_point.json",operating_point)


def _method_dir(config: dict[str, Any], method: str, seed: int) -> Path:
    return training_run(config, method, seed)


def _train_matrix(config: dict[str, Any], resume: bool) -> dict[tuple[str, int], Path]:
    pipeline = config.get("pipeline", {})
    methods = list(pipeline.get("methods", ["standalone", "vanilla_kd", "racmed"]))
    seeds = [int(seed) for seed in pipeline.get("seeds", config.get("seeds", [config.get("seed", 0)]))]
    artifacts: dict[tuple[str, int], Path] = {}
    total = len(methods) * len(seeds)
    current = 0
    for seed in seeds:
        per_seed: dict[str, Path] = {}
        for method in methods:
            current += 1
            run = deepcopy(config)
            run["seed"] = seed
            run["output_dir"] = str(_method_dir(config, method, seed))
            checkpoint = Path(run["output_dir"]) / "checkpoints" / "best.pt"
            fields = {"method": method, "seed": seed, "run": f"{current}/{total}"}
            if resume and checkpoint.exists():
                log_skip("train", **fields)
                summary=Path(run["output_dir"])/"summary.csv"
                if not _summary_has_current_reliability_schema(summary,str(config.get("task_type",""))):
                    with logged_stage("reevaluate_metrics_schema",**fields):
                        evaluate_checkpoint(run,checkpoint,method)
            else:
                with logged_stage("train", **fields):
                    train_method(run, method, per_seed)
            if not checkpoint.exists():
                raise RuntimeError(f"Training stage did not produce {checkpoint}")
            per_seed[method] = checkpoint
            artifacts[(method, seed)] = checkpoint
    return artifacts


def _train_variants(config: dict[str, Any], artifacts: dict[tuple[str, int], Path], resume: bool) -> None:
    """Run the predeclared, decision-relevant mechanism variants."""

    for variant in config.get("pipeline", {}).get("training_variants", []):
        name = str(variant["name"])
        method = str(variant.get("method", "racmed"))
        for seed in [int(value) for value in variant.get("seeds", [config.get("seed", 0)])]:
            run = merged(config, variant.get("overrides", {}))
            run["seed"] = seed
            run["output_dir"] = str(ablation_run(config, name, seed))
            run.setdefault("reporting", {})["method_label"] = name
            checkpoint = Path(run["output_dir"]) / "checkpoints" / "best.pt"
            if resume and checkpoint.exists():
                log_skip("training_variant", variant=name, seed=seed)
                summary=Path(run["output_dir"])/"summary.csv"
                if not _summary_has_current_reliability_schema(summary,str(config.get("task_type",""))):
                    with logged_stage("reevaluate_variant_metrics_schema",variant=name,seed=seed):
                        evaluate_checkpoint(run,checkpoint,name)
                continue
            per_seed = {candidate: path for (candidate, candidate_seed), path in artifacts.items() if candidate_seed == seed}
            with logged_stage("training_variant", variant=name, method=method, seed=seed):
                train_method(run, method, per_seed)


def _operating_point_sensitivity(
    config:dict[str,Any],artifacts:dict[tuple[str,int],Path],resume:bool
) -> None:
    """Evaluate selected segmentation models at a shared 0.5 operating point."""

    specification=config.get("pipeline",{}).get("fixed_threshold_sensitivity")
    if not specification or not str(config.get("task_type","")).startswith("segmentation"):
        return
    threshold=float(specification.get("threshold",.5))
    label=str(threshold).replace(".","_")
    methods=[str(value) for value in specification.get("methods",("standalone","vanilla_kd","racmed"))]
    seeds=[int(value) for value in specification.get("seeds",config.get("pipeline",{}).get("seeds",[0]))]
    for method in methods:
        for seed in seeds:
            checkpoint=artifacts.get((method,seed))
            if checkpoint is None:
                continue
            run=deepcopy(config);run["seed"]=seed
            run["output_dir"]=str(operating_point_run(config,f"fixed_{label}",method,seed))
            run["evaluation"]={**run.get("evaluation",{}),"segmentation_threshold_calibration":False,"fixed_segmentation_threshold":threshold}
            summary=Path(run["output_dir"])/"summary.csv"
            if resume and _summary_has_current_reliability_schema(summary,str(config.get("task_type",""))):
                log_skip("fixed_threshold_sensitivity",method=method,seed=seed,threshold=threshold)
                continue
            with logged_stage("fixed_threshold_sensitivity",method=method,seed=seed,threshold=threshold):
                evaluate_checkpoint(run,checkpoint,method)


def _compression_variants(config: dict[str, Any], artifacts: dict[tuple[str, int], Path], resume: bool) -> None:
    pipeline = config.get("pipeline", {})
    variants = list(pipeline.get("compression_variants", []))
    for variant in variants:
        name = str(variant["name"])
        method = str(variant.get("method", "racmed"))
        seeds=[int(value) for value in variant.get("seeds",[config.get("seed",0)])]
        export_seeds={int(value) for value in variant.get("export_seeds",seeds)}
        benchmark_seeds={int(value) for value in variant.get("benchmark_seeds",seeds)}
        for seed in seeds:
            run = merged(config, variant.get("overrides", {}))
            run["seed"] = seed
            run["output_dir"] = str(compression_run(config, name, seed))
            source = artifacts.get((method, seed))
            operating_point=_source_validation_operating_point(run,source)
            run.setdefault("reporting", {}).update({
                "comparison_family": "pruning",
                "compression_source_method": method,
                "compression_source_result_dir": str(source.parent.parent) if source is not None else None,
            })
            comp = Path(run["output_dir"]) / "compression"
            pruned = comp / "pruned.pt"
            if resume and pruned.exists():
                log_skip("prune", variant=name, seed=seed)
            else:
                with logged_stage("prune", variant=name, source=method, seed=seed):
                    run_pruning(run, source)
            summary = Path(run["output_dir"]) / "summary.csv"
            operating_point_marker=Path(run["output_dir"])/"compression_operating_point.json"
            if (
                resume
                and _summary_has_current_reliability_schema(summary,str(config.get("task_type","")))
                and _marker_matches(operating_point_marker,operating_point)
            ):
                log_skip("evaluate_pruned", variant=name, seed=seed)
            else:
                with logged_stage("evaluate_pruned", variant=name, seed=seed):
                    evaluate_checkpoint(run, pruned, name)
                    if operating_point is not None:
                        operating_point_marker.write_text(json.dumps(operating_point,indent=2))
            deploy = Path(run["output_dir"]) / "deployment"
            should_export = bool(variant.get("export", False)) and seed in export_seeds
            if should_export:
                deployment_operating_point=deploy/"compression_operating_point.json"
                evaluate_onnx=bool(variant.get("evaluate_onnx",True))
                export_qdq=bool(variant.get("tensorrt_qdq",False))
                if resume and _export_complete(
                    run,deploy,evaluate_onnx,export_qdq,operating_point
                ):
                    log_skip("quantize_export", variant=name, seed=seed)
                else:
                    with logged_stage("quantize_export", variant=name, seed=seed):
                        run_export(
                            run,
                            pruned,
                            evaluate_onnx=evaluate_onnx,
                            tensorrt_qdq=export_qdq,
                        )
                        if operating_point is not None:
                            deployment_operating_point.write_text(json.dumps(operating_point,indent=2))
            if bool(variant.get("benchmark", False)) and seed in benchmark_seeds:
                artifact_map=_artifact_manifest(deploy)
                marker=deploy/"benchmark_complete.json"
                if resume and _benchmark_marker_matches(marker,artifact_map):
                    log_skip("benchmark", variant=name, seed=seed)
                else:
                    with logged_stage("benchmark", variant=name, seed=seed, backend="pytorch+onnx_suite" if artifact_map else "pytorch"):
                        if artifact_map:
                            run_onnx_benchmark_suite(run,artifact_map,"auto")
                        energy_result=run_benchmark(run,None,"pytorch","auto",energy=True,output_label="pytorch_pruned")
                        pytorch_result=energy_result["latency"]
                        (deploy/"latency.json").write_text(json.dumps(pytorch_result,indent=2))
                        marker.write_text(json.dumps({
                            "pytorch":pytorch_result,
                            "onnx_variants":sorted(label for label in artifact_map if label.startswith("onnx_")),
                        },indent=2))


def _deployment_variants(
    config: dict[str, Any],
    artifacts: dict[tuple[str, int], Path],
    resume: bool,
) -> None:
    """Export declared trained checkpoints without adding a pruning stage."""

    for variant in config.get("pipeline", {}).get("deployment_variants", []):
        name=str(variant["name"])
        method=str(variant.get("method", "racmed"))
        evaluate_onnx=bool(variant.get("evaluate_onnx", True))
        export_qdq=bool(variant.get("tensorrt_qdq", False))
        for seed in [int(value) for value in variant.get("seeds", [config.get("seed", 0)])]:
            checkpoint=artifacts.get((method,seed))
            if checkpoint is None:
                raise RuntimeError(f"Deployment variant '{name}' requires {method} seed {seed}.")
            run=merged(config,variant.get("overrides",{}))
            run["seed"]=seed
            run["output_dir"]=str(checkpoint.parent.parent)
            deploy=Path(run["output_dir"])/"deployment"
            if resume and _export_complete(run,deploy,evaluate_onnx,export_qdq):
                log_skip("deployment_export",variant=name,method=method,seed=seed)
                continue
            with logged_stage("deployment_export",variant=name,method=method,seed=seed):
                run_export(
                    run,
                    checkpoint,
                    evaluate_onnx=evaluate_onnx,
                    tensorrt_qdq=export_qdq,
                )


def run_task(config: dict[str, Any], resume: bool = False) -> None:
    """Run teacher validation, model training, and configured local deployment stages."""

    pipeline = config.get("pipeline", {})
    name = str(config.get("experiment_name", config.get("dataset", {}).get("name", "task")))
    methods = pipeline.get("methods", [])
    seeds = pipeline.get("seeds", config.get("seeds", [config.get("seed", 0)]))
    pipeline_logger().info(
        "TASK  | name=%s task_type=%s methods=%d seeds=%d training_variants=%d compression_variants=%d deployment_variants=%d",
        name,
        config.get("task_type"),
        len(methods),
        len(seeds),
        len(pipeline.get("training_variants", [])),
        len(pipeline.get("compression_variants", [])),
        len(pipeline.get("deployment_variants", [])),
    )
    teacher_metrics = teacher_output_dir(config) / "metrics.json"
    if bool(pipeline.get("evaluate_teacher", True)):
        if resume and teacher_metrics.exists():
            log_skip("validate_teacher", task=name)
        else:
            with logged_stage("validate_teacher", task=name):
                evaluate_pretrained_teacher(config)
    comparison_dir=config.get("comparison_teacher_output_dir")
    if bool(pipeline.get("evaluate_teacher",True)) and comparison_dir and Path(comparison_dir)!=teacher_output_dir(config):
        matched_metrics=Path(comparison_dir)/"metrics.json"
        if resume and matched_metrics.exists():
            log_skip("validate_teacher_matched_test",task=name)
        else:
            with logged_stage("validate_teacher_matched_test",task=name):
                evaluate_pretrained_teacher(config,matched_test=True)
    artifacts = _train_matrix(config, resume)
    _train_variants(config, artifacts, resume)
    _operating_point_sensitivity(config,artifacts,resume)
    _deployment_variants(config,artifacts,resume)
    _compression_variants(config, artifacts, resume)


def run_experiments(config_paths: Iterable[str | Path], results_dir: str | Path, resume: bool = False) -> None:
    """Run all task configs and finish with canonical contradiction analysis."""

    paths = list(config_paths)
    logger = pipeline_logger()
    logger.info("PIPELINE | tasks=%d resume=%s results_dir=%s", len(paths), resume, results_dir)
    start = time.perf_counter()
    task_progress = progress_bar(paths, desc="RACMed tasks", total=len(paths), unit="task", leave=True)
    for index, path in enumerate(task_progress, start=1):
        config = load_config(path)
        name = config.get("experiment_name", Path(path).stem)
        task_progress.set_postfix(task=name)
        with logged_stage("task", task=name, position=f"{index}/{len(paths)}"):
            run_task(config, resume)
    with logged_stage("analyze_results", results_dir=results_dir):
        analyze_results(results_dir)
    logger.info("COMPLETE | tasks=%d elapsed=%.1fs", len(paths), time.perf_counter() - start)
