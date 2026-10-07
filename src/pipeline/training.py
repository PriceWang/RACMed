"""Training stage used by the numbered RACMed command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.engine.trainer import run_training
from src.pipeline.method_config import prepare_method_config
from src.pipeline.progress import pipeline_logger


KD_TYPES = {
    "vanilla_kd": "vanilla",
    "attention_kd": "attention",
    "relation_kd": "relation",
}
TRAINING_METHODS = ("standalone", "racmed", *KD_TYPES)


def train_method(
    config: dict[str, Any],
    method: str,
    artifact_paths: Mapping[str, str | Path] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Train one compressed-model objective from a shared task config."""

    if method not in TRAINING_METHODS:
        raise ValueError(f"Unknown training method: {method}")
    resolved = prepare_method_config(config, method, artifact_paths)
    if method == "standalone":
        result = run_training(resolved, "compressed", debug)
    elif method == "racmed":
        result = run_training(resolved, "racmed", debug)
    else:
        result = run_training(resolved, "kd", debug, KD_TYPES[method])
    metrics = result.get("test", {})
    primary = metrics.get("auc", metrics.get("dice", metrics.get("accuracy")))
    pipeline_logger().info(
        "TRAIN_RESULT | method=%s seed=%s primary_metric=%s",
        method,
        config.get("seed", 0),
        f"{primary:.4f}" if primary is not None else "n/a",
    )
    return result
