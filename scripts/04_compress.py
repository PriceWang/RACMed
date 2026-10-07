#!/usr/bin/env python
"""Pipeline 04: profile layer risk and apply configured pruning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cli import config_parser, parsed_config
from src.compression.flop_backfill import backfill_compression_flops
from src.pipeline.pruning import run_pruning
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backfill-flops",action="store_true",help="Populate FLOPs for all persisted compression variants in config.output_dir.")
    parser.add_argument("--force",action="store_true",help="Recompute existing FLOP records.")
    args = parser.parse_args()
    config = parsed_config(args)
    configure_pipeline_logging(Path(config.get("output_dir", "outputs/experiment")) / "logs/pipeline.log")
    stage="backfill_flops" if args.backfill_flops else "prune"
    with logged_stage(stage, seed=config.get("seed", 0)):
        if args.backfill_flops:backfill_compression_flops(config.get("output_dir","outputs/experiment"),args.force)
        else:run_pruning(config, args.checkpoint, args.debug)
