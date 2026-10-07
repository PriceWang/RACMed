"""Seed, package, platform, and git provenance capture."""

from __future__ import annotations
import importlib.metadata, json, platform, subprocess
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed:int,deterministic:bool=True)->None:
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if deterministic: torch.use_deterministic_algorithms(True,warn_only=True)


def save_provenance(directory:str|Path)->None:
    directory=Path(directory); versions={name:_version(name) for name in ("torch","numpy","scipy","pandas","monai","timm","onnxruntime")}
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit=None
    (directory/"package_versions.json").write_text(json.dumps({"python":platform.python_version(),"packages":versions,"git_commit":commit},indent=2))


def _version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None
