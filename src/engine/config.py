"""YAML loading and dotted command-line overrides."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle: data=yaml.safe_load(handle) or {}
    return data


def _value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for override in overrides or []:
        if "=" not in override: raise ValueError(f"Override must be key=value: {override}")
        key,value=override.split("=",1); cursor=config
        parts=key.split(".")
        for part in parts[:-1]: cursor=cursor.setdefault(part,{})
        cursor[parts[-1]]=_value(value)
    return config


def merged(base: dict, child: dict) -> dict:
    out=copy.deepcopy(base)
    for key,value in child.items(): out[key]=merged(out.get(key,{}),value) if isinstance(value,dict) and isinstance(out.get(key),dict) else copy.deepcopy(value)
    return out
