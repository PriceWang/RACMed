"""Utilities for relabeling evaluation artifacts without changing metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def relabel_experiment_outputs(output_dir: str | Path, method_label: str) -> None:
    """Set a public method label in summary and metrics artifacts.

    This is used for post-training evaluations such as pruned or quantized
    checkpoints, where the loaded model is the compressed RACMed target but the
    reported method should describe the deployment/compression variant.
    """

    directory = Path(output_dir)
    summary = directory / "summary.csv"
    if summary.exists():
        frame = pd.read_csv(summary)
        frame["method"] = method_label
        frame.to_csv(summary, index=False)

    metrics = directory / "metrics.json"
    if metrics.exists():
        payload = json.loads(metrics.read_text())
        payload["method"] = method_label
        metrics.write_text(json.dumps(payload, indent=2))
