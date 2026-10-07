#!/usr/bin/env python
"""Pipeline 07: compute cross-task contradiction and statistical result CSVs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.analysis import analyze_results
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--epsilon", type=float, default=0.01, help="A-priori negligible-drop threshold.")
    parser.add_argument("--tau", type=float, default=0.05, help="A-priori clinically material-drop threshold.")
    args = parser.parse_args()
    configure_pipeline_logging(Path(args.results_dir) / "pipeline.log")
    with logged_stage("analyze_results", results_dir=args.results_dir):
        analyze_results(args.results_dir, args.epsilon, args.tau)
