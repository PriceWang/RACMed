"""Canonical public names for RACMed experiment methods."""

from __future__ import annotations

import pandas as pd


def public_method_name(mode: str | None, kd_type: str | None = None) -> str:
    """Return the unambiguous method name used in reports and CSV artifacts.

    Public outputs use ``standalone_compressed`` for the compact model trained
    without knowledge distillation.
    """

    raw = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"compressed", "standalone", "standalone_compressed"}:
        return "standalone_compressed"
    if raw in {"vanilla_kd", "attention_kd", "relation_kd"}:
        return raw
    if raw == "uniform_magnitude_20_crosstask":
        return "uniform_magnitude_20_crosstask"
    if raw == "kd":
        kind = str(kd_type or "vanilla").strip().lower()
        if kind not in {"vanilla", "attention", "relation"}:
            raise ValueError(f"Unsupported retained KD baseline: {kind}")
        return kind if kind.endswith("_kd") else f"{kind}_kd"
    return raw


def normalize_method_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize method labels in result dataframes."""

    if frame.empty or "method" not in frame:
        return frame
    normalized = frame.copy()
    kd_types = normalized["kd_type"] if "kd_type" in normalized else [None] * len(normalized)
    normalized["method"] = [
        public_method_name(method, None if pd.isna(kd_type) else kd_type)
        for method, kd_type in zip(normalized["method"], kd_types)
    ]
    return normalized
