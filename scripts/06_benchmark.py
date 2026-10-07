#!/usr/bin/env python
"""Pipeline 06: benchmark a PyTorch, ONNX Runtime, TensorRT, or CoreML artifact."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.benchmarking import run_benchmark
from src.pipeline.cli import config_parser, parsed_config
from src.pipeline.progress import configure_pipeline_logging, logged_stage


if __name__ == "__main__":
    parser = config_parser(__doc__)
    parser.add_argument("--artifact")
    parser.add_argument("--backend", choices=("auto", "pytorch", "onnx", "tensorrt", "coreml"), default="auto")
    parser.add_argument("--target", choices=("auto", "raspi5", "jetson-cpu", "jetson-cuda", "apple-m4"), default="auto")
    parser.add_argument("--onnx-provider", choices=("auto", "cpu", "cuda", "tensorrt"), default="auto")
    parser.add_argument("--output-label", help="Suffix for distinct latency/energy result files.")
    parser.add_argument("--energy", action="store_true")
    parser.add_argument(
        "--energy-only",action="store_true",
        help="Sample workload power and update energy JSON without rerunning latency timing.",
    )
    parser.add_argument(
        "--memory-only",action="store_true",
        help="Run one inference and add process/accelerator peak memory to an existing latency record.",
    )
    args = parser.parse_args()
    config = parsed_config(args)
    configure_pipeline_logging(Path(config.get("output_dir", "outputs/experiment")) / "logs/pipeline.log")
    with logged_stage("benchmark", backend=args.backend, target=args.target):
        result = run_benchmark(
            config,args.artifact,args.backend,args.target,args.energy,
            output_label=args.output_label,onnx_provider=args.onnx_provider,
            energy_only=args.energy_only,memory_only=args.memory_only,
        )
    print(json.dumps(result, indent=2), flush=True)
