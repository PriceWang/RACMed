"""Model path resolution without storing machine-local absolute paths in configs."""

from __future__ import annotations

import os
from pathlib import Path


MODEL_ROOT_ENV = "RACMED_MODEL_ROOT"


def resolve_model_path(path: str | Path | None, default_relative: str | None = None) -> Path:
    """Expand environment variables and resolve model paths.

    If ``path`` is omitted, ``default_relative`` is resolved under
    ``$RACMED_MODEL_ROOT`` when set, otherwise under the repository-local
    ``external_models`` directory.
    """

    if path:
        raw = os.fspath(path)
        if MODEL_ROOT_ENV not in os.environ:
            for token in (f"${{{MODEL_ROOT_ENV}}}", f"${MODEL_ROOT_ENV}"):
                if token in raw:
                    suffix = raw.replace(token, "").lstrip("/\\")
                    return (Path("external_models") / suffix).expanduser()
        return Path(os.path.expandvars(raw)).expanduser()
    base = Path(os.environ.get(MODEL_ROOT_ENV, "external_models")).expanduser()
    return base / default_relative if default_relative else base
