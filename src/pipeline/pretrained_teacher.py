"""Evaluation utilities for externally pretrained teacher models."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.datasets.dataset_factory import create_dataloaders
from src.engine.evaluator import evaluate
from src.engine.reproducibility import save_provenance, seed_everything
from src.models.model_factory import create_model
from src.pipeline.progress import pipeline_logger
from src.pipeline.layout import task_root


def _metric_value(metrics: Any, path: str) -> float:
    value=metrics
    for part in path.split("."):
        value=value[int(part)] if isinstance(value,list) else value[part]
    return float(value)


def evaluate_reproduction_gates(
    metrics: dict[str,Any],gates:list[dict[str,Any]],enforce:bool=False
) -> dict[str,Any]:
    """Evaluate configured minimum teacher-reproduction metrics."""
    rows=[]
    for gate in gates:
        metric=str(gate["metric"]);minimum=float(gate["minimum"]);actual=_metric_value(metrics,metric)
        rows.append({"metric":metric,"actual":actual,"minimum":minimum,"passed":actual>=minimum})
    result={"passed":all(row["passed"] for row in rows),"gates":rows}
    if enforce and not result["passed"]:
        failures=", ".join(f"{row['metric']}={row['actual']:.4f} < {row['minimum']:.4f}" for row in rows if not row["passed"])
        raise RuntimeError(f"External teacher reproduction gate failed: {failures}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def teacher_output_dir(config: dict) -> Path:
    """Return the configured teacher-evaluation output directory."""

    if config.get("teacher_output_dir"):
        return Path(config["teacher_output_dir"])
    return task_root(config) / "teacher" / "native"


def teacher_evaluation_config(config: dict) -> dict:
    """Return a teacher-evaluation config without mutating training settings."""

    eval_config = deepcopy(config)
    teacher_eval = eval_config.get("teacher_evaluation", {})
    if "dataset_image_size" in teacher_eval:
        eval_config.setdefault("dataset", {})["image_size"] = teacher_eval["dataset_image_size"]
    if "dataset_split_policy" in teacher_eval:
        eval_config.setdefault("dataset", {})["split_policy"] = teacher_eval["dataset_split_policy"]
    if "max_eval_batches" in teacher_eval:
        eval_config.setdefault("training", {})["max_eval_batches"] = teacher_eval["max_eval_batches"]
    return eval_config


def evaluate_pretrained_teacher(
    config: dict,
    debug: bool = False,
    matched_test: bool = False,
) -> dict[str, Any]:
    """Evaluate an external teacher in exact-replay or matched-test mode."""

    original=deepcopy(config)
    config = deepcopy(config) if matched_test else teacher_evaluation_config(config)
    seed_everything(int(config.get("seed", 0)))
    if matched_test:
        configured=original.get("comparison_teacher_output_dir")
        if not configured:raise ValueError("matched_test teacher evaluation requires comparison_teacher_output_dir")
        output=Path(configured)
    else:
        output = teacher_output_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_provenance(output)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    task = config.get("task_type", config.get("model", {}).get("task_type", "classification"))
    classes = int(config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 2)))
    pipeline_logger().info("TEACHER_SETUP | step=create_dataloaders dataset=%s",config.get("dataset",{}).get("name"))
    loaders = create_dataloaders(config, debug)
    pipeline_logger().info("TEACHER_SETUP | step=load_model model=%s",config.get("teacher",{}).get("name"))
    model = create_model(config, "teacher").to(device)
    pipeline_logger().info("TEACHER_SETUP | step=ready device=%s",device)
    max_eval_batches = config.get("training", {}).get("max_eval_batches")
    eval_cfg = config.get("evaluation", {})
    threshold_strategy = eval_cfg.get("threshold_strategy") if task == "classification" else None
    evaluation_split="test" if matched_test else str(config.get("teacher_evaluation",{}).get("split","test"))
    if evaluation_split not in loaders:raise ValueError(f"Unknown teacher evaluation split: {evaluation_split}")
    pipeline_logger().info(
        "TEACHER | model=%s dataset=%s split=%s cases=%d device=%s",
        config.get("teacher", {}).get("name"),
        config.get("dataset", {}).get("name"),
        evaluation_split,
        len(loaders[evaluation_split].dataset),
        device,
    )
    metrics, cases = evaluate(
        model,
        loaders[evaluation_split],
        task,
        device,
        classes,
        max_batches=max_eval_batches,
        threshold_strategy=threshold_strategy,
        sliding_window=eval_cfg.get("sliding_window"),
        progress_label=f"teacher:{config.get('dataset', {}).get('name', task)}",
        small_component_threshold=int(eval_cfg.get("small_component_threshold",config.get("risk",{}).get("small_component_threshold",64))),
    )

    payload = {"test": metrics, "mode": "teacher", "method": "teacher", "pretrained_teacher": True,
               "evaluation_protocol":"matched_test" if matched_test else "exact_replay"}
    (output / "metrics.json").write_text(json.dumps(_jsonable(payload), indent=2))
    pd.DataFrame(cases).to_csv(output / "per_case_metrics.csv", index=False)
    flat = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    pd.DataFrame(
        [
            {
                **flat,
                "method": "teacher",
                "mode": "teacher",
                "kd_type": None,
                "task": task,
                "seed": config.get("seed", 0),
                "score": flat.get("accuracy", flat.get("dice", 0.0)),
            }
        ]
    ).to_csv(output / "summary.csv", index=False)
    gates=[] if matched_test else config.get("teacher_evaluation",{}).get("gates",[])
    if gates:
        gate_result=evaluate_reproduction_gates(metrics,gates,enforce=False)
        (output/"reproduction_gate.json").write_text(json.dumps(gate_result,indent=2))
        if not gate_result["passed"] and bool(config.get("teacher_evaluation",{}).get("enforce_gate",True)):
            evaluate_reproduction_gates(metrics,gates,enforce=True)
        pipeline_logger().info("TEACHER_GATE | passed=%s checks=%d", gate_result["passed"], len(gates))
    primary = metrics.get("auc", metrics.get("dice", metrics.get("accuracy")))
    pipeline_logger().info("TEACHER_RESULT | primary_metric=%s cases=%d", f"{primary:.4f}" if primary is not None else "n/a", len(cases))
    return {"test": metrics, "output_dir": str(output)}
