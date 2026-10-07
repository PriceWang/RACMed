"""Shared command-line parsing for RACMed pipeline scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.engine.config import apply_overrides, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def config_parser(description: str) -> argparse.ArgumentParser:
    """Return the common parser used by public pipeline commands."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, e.g. training.lr=1e-4.")
    parser.add_argument("--debug", action="store_true", help="Use the synthetic debug dataset and short execution limits.")
    parser.add_argument("overrides", nargs="*", help=argparse.SUPPRESS)
    return parser


def parsed_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load YAML and apply both explicit and positional dotted overrides."""
    overrides = list(args.override) + list(args.overrides)
    return apply_overrides(load_config(args.config), overrides)
