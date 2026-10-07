#!/usr/bin/env python
"""Pipeline 05: quantize and export ONNX, CoreML, or TensorRT artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cli import config_parser, parsed_config
from src.pipeline.exporting import run_export
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--dynamic-axes", action="store_true")
    parser.add_argument("--evaluate-onnx", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument(
        "--coreml-only",action="store_true",
        help="Export only CoreML without regenerating canonical ONNX artifacts.",
    )
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--tensorrt-qdq", action="store_true", help="Export calibrated Q/DQ INT8 graphs for TensorRT.")
    args = parser.parse_args()
    config = parsed_config(args)
    configure_pipeline_logging(Path(config.get("output_dir", "outputs/experiment")) / "logs/pipeline.log")
    with logged_stage("quantize_export", seed=config.get("seed", 0)):
        run_export(
            config,checkpoint=args.checkpoint,output=args.output,dynamic_axes=args.dynamic_axes,
            evaluate_onnx=args.evaluate_onnx,coreml=args.coreml,tensorrt=args.tensorrt,
            debug=args.debug,tensorrt_qdq=args.tensorrt_qdq,coreml_only=args.coreml_only,
        )
