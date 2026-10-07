"""Console/file logging setup."""

import logging
from pathlib import Path

from src.utils.progress import TqdmLoggingHandler


def configure_logger(output_dir):
    path=Path(output_dir)/"logs";path.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[TqdmLoggingHandler(),logging.FileHandler(path/"train.log")],force=True)
    return logging.getLogger("racmed")
