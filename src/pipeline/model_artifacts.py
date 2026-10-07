"""Helpers for deployment/compression scripts that operate on configurable model roles."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def pipeline_model_role(config: dict[str, Any], section: str) -> str:
    """Return the model role selected for a pipeline section.

    ``compressed_model`` is the default. Pretrained compression configs can
    set ``compression.model_role: teacher``
    or ``deployment.model_role: teacher`` to operate directly on a local bundle.
    """

    return str(config.get(section, {}).get("model_role") or config.get("deployment", {}).get("model_role") or "compressed_model")


def pipeline_checkpoint(
    config: dict[str, Any],
    output_dir: str | Path,
    section: str,
    explicit: str | Path | None = None,
) -> Path | None:
    """Resolve the checkpoint used by a compression or deployment stage.

    Relative configured paths are resolved inside the experiment directory.
    Deployment prefers the pruned artifact so teacher-role compression is not
    accidentally benchmarked using the original external weights.
    """

    output_dir = Path(output_dir)
    if explicit:
        return Path(explicit)
    selected = config.get(section, {}).get("checkpoint")
    if selected:
        path = Path(selected)
        return path if path.is_absolute() else output_dir / path
    candidates = []
    if section == "deployment":
        candidates.append(output_dir / "compression" / "pruned.pt")
    candidates.append(output_dir / "checkpoints" / "best.pt")
    return next((path for path in candidates if path.exists()), None)


def _role_config(config: dict[str, Any], role: str) -> dict[str, Any]:
    base = dict(config.get("model", {}))
    base.update(config.get(role, {}))
    return base


def example_input_shape(config: dict[str, Any], role: str = "compressed_model") -> list[int]:
    """Build an example input shape from deployment, image_size, or crop_size."""

    deployment = config.get("deployment", {})
    dataset = config.get("dataset", {})
    role_cfg = _role_config(config, role)
    spatial = deployment.get("input_size") or dataset.get("image_size") or dataset.get("crop_size") or [32, 32]
    if spatial is None:
        spatial = [32, 32]
    return [1, int(role_cfg.get("in_channels", 1)), *[int(value) for value in spatial]]


def export_config(config: dict[str, Any], role: str) -> dict[str, Any]:
    """Return a copy configured for direct model tracing/export.

    MONAI bundle adapters may use sliding-window inference for full-volume
    evaluation. ONNX tracing should use a direct 96³ crop forward instead.
    """

    out = copy.deepcopy(config)
    role_cfg = out.get(role)
    if isinstance(role_cfg, dict) and role_cfg.get("source") == "monai_bundle":
        role_cfg["sliding_window"] = False
    return out
