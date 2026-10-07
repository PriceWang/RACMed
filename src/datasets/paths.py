"""Dataset root resolution without hard-coding site-local absolute paths."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "RACMED_DATA_ROOT"

_ALIASES: dict[str, tuple[str, ...]] = {
    "acdc": ("acdc",),
    "brats2018": ("brats2018",),
    "rsna": ("rsna",),
    "rsna_pneumonia": ("rsna",),
}


def _expand_env(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.fspath(path))).expanduser()


def resolve_dataset_root(root: str | Path, dataset_name: str | None = None) -> Path:
    """Resolve a configured dataset root.

    The resolver supports ``${RACMED_DATA_ROOT}`` placeholders and common
    official-folder aliases for the validated task families. It returns an
    existing path when possible and raises an actionable error otherwise.
    """

    configured = _expand_env(root)
    candidates: list[Path] = [configured]

    env_root = os.environ.get(DATA_ROOT_ENV)
    aliases = _ALIASES.get((dataset_name or "").lower(), ())

    if env_root:
        env_base = Path(env_root).expanduser()
        for alias in aliases:
            candidates.append(env_base / alias)
        if not configured.is_absolute():
            candidates.append(env_base / configured)
        if configured.is_absolute():
            for alias in aliases:
                candidates.append(configured.parent / alias)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Dataset root for {dataset_name or 'dataset'} was not found. "
        f"Set {DATA_ROOT_ENV}, use a relative project data path, or override dataset.root. "
        "Resolved local paths are intentionally omitted from this error."
    )
