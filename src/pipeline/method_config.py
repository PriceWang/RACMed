"""Resolve method-specific training stages from a shared task configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.engine.config import merged


def prepare_method_config(
    config: dict[str, Any],
    method: str,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Merge a method override and resolve its optional initialization checkpoint.

    ``artifact_paths`` is populated by the full pipeline after earlier stages.
    Direct single-method commands may instead provide
    ``compressed_model.checkpoint`` through a command-line override.
    """

    method_settings = dict(config.get("method_overrides", {}).get(method, {}))
    initialize_from = method_settings.pop("initialize_from", None)
    result = merged(config, method_settings)
    if not initialize_from:
        return result
    if artifact_paths and initialize_from in artifact_paths:
        result.setdefault("compressed_model", {})["checkpoint"] = str(
            artifact_paths[initialize_from]
        )
        return result
    checkpoint = result.get("compressed_model", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError(
            f"Method '{method}' initializes from '{initialize_from}'. Run that stage first "
            "or override compressed_model.checkpoint with its best checkpoint."
        )
    return result
