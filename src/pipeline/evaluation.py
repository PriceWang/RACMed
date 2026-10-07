"""Checkpoint evaluation stage shared by pipeline commands and orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.engine.checkpoint import load_model_checkpoint
from src.engine.trainer import Trainer
from src.pipeline.model_artifacts import pipeline_checkpoint, pipeline_model_role
from src.pipeline.progress import pipeline_logger
from src.pipeline.result_labels import relabel_experiment_outputs


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint: str | Path | None = None,
    method_label: str | None = None,
    model_role: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Evaluate one configured model role and persist canonical metrics."""

    cfg = dict(config)
    cfg["evaluation"] = dict(config.get("evaluation", {}))
    if model_role:
        cfg["evaluation"]["model_role"] = model_role
    role = cfg["evaluation"].get("model_role") or pipeline_model_role(cfg, "evaluation")
    if role != "compressed_model":
        cfg["compressed_model"] = {**cfg.get("model", {}), **cfg.get(role, {})}
    trainer = Trainer(cfg, "compressed", debug)
    selected = pipeline_checkpoint(cfg, cfg.get("output_dir", "outputs/experiment"), "evaluation", checkpoint)
    if selected is not None:
        if not selected.exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {selected}")
        trainer.model,_=load_model_checkpoint(selected,trainer.model)
        trainer.model=trainer.model.to(trainer.device)
    metrics = trainer.test()
    if method_label:
        relabel_experiment_outputs(cfg.get("output_dir", "outputs/experiment"), method_label)
    primary = metrics.get("auc", metrics.get("dice", metrics.get("accuracy")))
    pipeline_logger().info(
        "EVALUATION_RESULT | method=%s primary_metric=%s",
        method_label or role,
        f"{primary:.4f}" if primary is not None else "n/a",
    )
    return metrics
