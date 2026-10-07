#!/usr/bin/env python
"""Pipeline 03: evaluate a saved RACMed or external-teacher checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cli import config_parser, parsed_config
from src.pipeline.evaluation import evaluate_checkpoint
from src.pipeline.lesion_reliability import evaluate_lesion_operating_curve
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--method-label")
    parser.add_argument("--model-role")
    parser.add_argument("--lesion-operating-curve",action="store_true",help="Evaluate FROC and matched-FP lesion recall without changing canonical metrics.")
    args = parser.parse_args()
    config = parsed_config(args)
    configure_pipeline_logging(Path(config.get("output_dir", "outputs/experiment")) / "logs/pipeline.log")
    with logged_stage("evaluate", method=args.method_label, seed=config.get("seed", 0)):
        if args.lesion_operating_curve:
            evaluate_lesion_operating_curve(config,args.checkpoint,args.method_label,args.model_role,args.debug)
        else:evaluate_checkpoint(config, args.checkpoint, args.method_label, args.model_role, args.debug)
