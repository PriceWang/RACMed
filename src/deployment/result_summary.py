"""Aggregate reproducible deployment and model-complexity measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _config(result_dir: Path) -> dict[str, Any]:
    path=result_dir/"config.yaml"
    return yaml.safe_load(path.read_text()) if path.is_file() else {}


def _identity(root: Path, result_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    relative=result_dir.relative_to(root)
    parts=relative.parts
    category=next((name for name in ("compression","compressed") if name in parts),None)
    variant=parts[parts.index(category)+1] if category else None
    seed_match=re.fullmatch(r"seed_(\d+)",result_dir.name)
    return {
        "experiment":parts[0] if parts else result_dir.name,
        "task":config.get("task_type"),
        "dataset":config.get("dataset",{}).get("name"),
        "compression_variant":variant,
        "seed":int(seed_match.group(1)) if seed_match else config.get("seed"),
        "result_dir":relative.as_posix(),
    }


def _variant_label(path: Path, target: str) -> tuple[str, int | None]:
    label=path.stem.removeprefix("latency_")
    normalized_target=target.replace("-","_")
    if normalized_target and label.endswith("_"+normalized_target):
        label=label[:-(len(normalized_target)+1)]
    match=re.search(r"_rep(\d+)$",label)
    repetition=int(match.group(1)) if match else None
    if match:label=label[:match.start()]
    return label,repetition


def aggregate_deployment_metrics(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write raw and mean±SD summaries for latency/RSS and FLOP records."""

    root=Path(results_dir);analysis=root/"analysis";analysis.mkdir(parents=True,exist_ok=True)
    hardware_rows=[]
    for path in sorted(root.glob("**/deployment/latency_*.json")):
        try:payload=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if "latency_mean_ms" not in payload:continue
        result_dir=path.parent.parent;config=_config(result_dir)
        target=str(payload.get("target","auto"));variant,repetition=_variant_label(path,target)
        hardware_rows.append({
            **_identity(root,result_dir,config),
            "backend_variant":variant,
            "repetition":repetition,
            "backend":payload.get("backend"),
            "target":target,
            "is_real_device":target not in {"","auto"},
            **{key:payload.get(key) for key in (
                "latency_mean_ms","latency_std_ms","latency_p50_ms","latency_p95_ms",
                "throughput_fps","peak_memory_mb","peak_memory_kind",
                "process_baseline_rss_mb","process_peak_rss_mb","process_incremental_peak_rss_mb",
            )},
        })
    hardware=pd.DataFrame(hardware_rows)
    hardware.to_csv(analysis/"deployment_hardware_records.csv",index=False)
    group_columns=[
        "experiment","task","dataset","compression_variant","seed",
        "target","backend_variant","backend","is_real_device",
    ]
    numeric=[
        "latency_mean_ms","latency_p50_ms","latency_p95_ms","throughput_fps",
        "peak_memory_mb","process_peak_rss_mb","process_incremental_peak_rss_mb",
    ]
    if hardware.empty:
        hardware_statistics=pd.DataFrame()
    else:
        hardware_statistics=(
            hardware.groupby(group_columns,dropna=False)[numeric]
            .agg(["mean","std","count"])
        )
        hardware_statistics.columns=[f"{metric}_{stat}" for metric,stat in hardware_statistics.columns]
        hardware_statistics=hardware_statistics.reset_index()
    hardware_statistics.to_csv(analysis/"deployment_hardware_statistics.csv",index=False)

    complexity_rows=[]
    for path in sorted(root.glob("**/compression/compression_metrics.json")):
        try:payload=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        result_dir=path.parent.parent;config=_config(result_dir)
        baseline_flops=payload.get("baseline_flops");flops=payload.get("flops")
        complexity_rows.append({
            **_identity(root,result_dir,config),
            "pruning_method":payload.get("pruning_method"),
            "params":payload.get("params"),
            "effective_params":payload.get("effective_params"),
            "physical_parameter_reduction":payload.get("physical_parameter_reduction"),
            "baseline_flops":baseline_flops,
            "flops":flops,
            "baseline_gflops":baseline_flops/1e9 if baseline_flops is not None else None,
            "gflops":flops/1e9 if flops is not None else None,
            "flops_reduction":payload.get("flops_reduction"),
            "model_size_mb":payload.get("model_size_mb"),
        })
    complexity=pd.DataFrame(complexity_rows)
    complexity.to_csv(analysis/"compression_complexity_records.csv",index=False)
    complexity_numeric=[
        "params","effective_params","physical_parameter_reduction","baseline_gflops",
        "gflops","flops_reduction","model_size_mb",
    ]
    complexity_groups=["experiment","task","dataset","compression_variant","pruning_method"]
    if complexity.empty:
        complexity_statistics=pd.DataFrame()
    else:
        complexity_statistics=(
            complexity.groupby(complexity_groups,dropna=False)[complexity_numeric]
            .agg(["mean","std","count"])
        )
        complexity_statistics.columns=[f"{metric}_{stat}" for metric,stat in complexity_statistics.columns]
        complexity_statistics=complexity_statistics.reset_index()
    complexity_statistics.to_csv(analysis/"compression_complexity_statistics.csv",index=False)
    return {
        "hardware_records":hardware,
        "hardware_statistics":hardware_statistics,
        "complexity_records":complexity,
        "complexity_statistics":complexity_statistics,
    }
