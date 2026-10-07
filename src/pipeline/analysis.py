"""Canonical cross-task contradiction analysis stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.contradiction.analyzer import CompressionContradictionAnalyzer
from src.contradiction.result_loader import load_canonical_results
from src.utils.method_naming import normalize_method_names
from src.pipeline.progress import pipeline_logger
from src.pipeline.lesion_reliability import aggregate_lesion_operating_curves
from src.deployment.result_summary import aggregate_deployment_metrics
from src.pipeline.result_freeze import freeze_results


def analyze_results(results_dir: str | Path, epsilon: float = 0.01, tau: float = 0.05) -> dict[str, Any]:
    """Load all canonical summaries and write analysis CSVs."""

    root = Path(results_dir)
    frame = normalize_method_names(load_canonical_results(root))
    if frame.empty:
        raise FileNotFoundError("No summary.csv files found. Run experiments first.")
    pipeline_logger().info(
        "ANALYSIS_INPUT | rows=%d methods=%d tasks=%d epsilon=%.3f tau=%.3f",
        len(frame),
        frame["method"].nunique(),
        frame["task"].nunique(),
        epsilon,
        tau,
    )
    result = CompressionContradictionAnalyzer(epsilon, tau).analyze(frame, root / "analysis")
    lesion_statistics=aggregate_lesion_operating_curves(root)
    result["lesion_operating_curve_statistics"]=lesion_statistics
    result.update(aggregate_deployment_metrics(root))
    manifest = freeze_results(root)
    result["result_manifest"] = manifest
    pipeline_logger().info("RESULT_FREEZE | status=%s artifacts=%d privacy=%s", manifest["status"], manifest["integrity"]["result_artifact_count"], manifest["privacy"]["passed"])
    contradictions = result["contradictions"]
    flagged = int(contradictions["is_contradiction"].sum()) if not contradictions.empty else 0
    pipeline_logger().info("ANALYSIS_RESULT | methods=%d contradictions=%d", len(contradictions), flagged)
    return result
