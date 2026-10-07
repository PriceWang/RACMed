"""Consistent tqdm progress bars for long RACMed loops."""

from __future__ import annotations

import os
import logging
import sys
from typing import Iterable, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting an active tqdm progress bar."""

    def __init__(self, stream=None) -> None:
        super().__init__(stream or sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = False,
):
    """Wrap an iterable with a low-noise, environment-controllable progress bar."""

    disabled = os.environ.get("RACMED_DISABLE_TQDM", "0").lower() in {"1", "true", "yes"}
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=disabled,
        file=sys.stdout,
    )
