"""Canonical filesystem layout for reproducible RACMed experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("outputs/experiments")


def task_root(config: dict[str, Any]) -> Path:
    """Return the configured root for one dataset/task experiment."""

    return Path(config.get("output_dir", DEFAULT_RESULTS_ROOT / "task"))


def training_run(config: dict[str, Any], method: str, seed: int) -> Path:
    """Return the run directory for a trained method and seed."""

    return task_root(config) / "training" / str(method) / f"seed_{int(seed)}"


def ablation_run(config: dict[str, Any], variant: str, seed: int) -> Path:
    """Return the run directory for a mechanism ablation."""

    return task_root(config) / "ablations" / str(variant) / f"seed_{int(seed)}"


def operating_point_run(
    config: dict[str, Any], label: str, method: str, seed: int
) -> Path:
    """Return the run directory for a fixed operating-point evaluation."""

    return (
        task_root(config)
        / "evaluation"
        / "operating_points"
        / str(label)
        / str(method)
        / f"seed_{int(seed)}"
    )


def compression_run(config: dict[str, Any], variant: str, seed: int) -> Path:
    """Return the run directory for a compression policy and seed."""

    return task_root(config) / "compression" / str(variant) / f"seed_{int(seed)}"
