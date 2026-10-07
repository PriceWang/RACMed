#!/usr/bin/env python
"""Pipeline 02: train a standalone, KD baseline, or RACMed model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cli import config_parser, parsed_config
from src.pipeline.layout import training_run
from src.pipeline.training import TRAINING_METHODS, train_method
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--method", choices=TRAINING_METHODS, default="racmed")
    args = parser.parse_args()
    config = parsed_config(args)
    # Direct method runs share the task-level output root in the YAML. Route
    # them exactly like the full pipeline so repeated seeds cannot overwrite
    # one another.
    seed = int(config.get("seed", 0))
    output = Path(config.get("output_dir", "outputs/experiment"))
    if output.name != f"seed_{seed}":
        config["output_dir"] = str(training_run(config, args.method, seed))
    configure_pipeline_logging(Path(config.get("output_dir", "outputs/experiment")) / "logs/pipeline.log")
    with logged_stage("train", method=args.method, seed=config.get("seed", 0)):
        train_method(config, args.method, debug=args.debug)
