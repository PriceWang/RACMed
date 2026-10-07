#!/usr/bin/env python
"""Pipeline 08: run the full validated multi-task experiment matrix."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.experiment_runner import run_experiments
from src.pipeline.progress import configure_pipeline_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, dest="configs")
    parser.add_argument("--results-dir", default="outputs/experiments")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_pipeline_logging(Path(args.results_dir) / "pipeline.log")
    run_experiments(args.configs, args.results_dir, args.resume)
