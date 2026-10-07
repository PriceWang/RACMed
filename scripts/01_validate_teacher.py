#!/usr/bin/env python
"""Pipeline 01: validate an external pretrained teacher on its native task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cli import config_parser, parsed_config
from src.pipeline.pretrained_teacher import evaluate_pretrained_teacher,teacher_output_dir
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--matched-test", action="store_true", help="Evaluate on the student test split for paired comparison.")
    args = parser.parse_args()
    config = parsed_config(args)
    log_root=(
        Path(config["comparison_teacher_output_dir"])
        if args.matched_test
        else teacher_output_dir(config)
    )
    configure_pipeline_logging(log_root / "logs/pipeline.log")
    with logged_stage("validate_teacher", task=config.get("experiment_name")):
        evaluate_pretrained_teacher(config, args.debug, args.matched_test)
