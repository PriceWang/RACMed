"""Unbuffered console and file progress logging for long experiment pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sys
import time
from typing import Iterator

from src.utils.progress import TqdmLoggingHandler


LOGGER_NAME = "racmed.pipeline"


def configure_pipeline_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Configure a dedicated logger that survives Trainer root-logger resets."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | PIPELINE | %(message)s")
    console = TqdmLoggingHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def pipeline_logger() -> logging.Logger:
    """Return the dedicated progress logger, configuring console output on demand."""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        configure_pipeline_logging()
    return logger


def _fields(values: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items() if value is not None)


@contextmanager
def logged_stage(stage: str, **fields: object) -> Iterator[None]:
    """Log START/DONE/FAIL events and elapsed wall time for one stage."""

    logger = pipeline_logger()
    details = _fields(fields)
    label = f"{stage} | {details}" if details else stage
    start = time.perf_counter()
    logger.info("START | %s", label)
    try:
        yield
    except Exception:
        logger.exception("FAIL  | %s | elapsed=%.1fs", label, time.perf_counter() - start)
        raise
    logger.info("DONE  | %s | elapsed=%.1fs", label, time.perf_counter() - start)


def log_skip(stage: str, **fields: object) -> None:
    """Log that a resumable artifact allowed a stage to be skipped."""

    details = _fields(fields)
    pipeline_logger().info("SKIP  | %s%s", stage, f" | {details}" if details else "")
