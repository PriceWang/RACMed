"""Reproducible FLOP profiling for persisted compression artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.compression.compression_metrics import estimate_flops
from src.engine.checkpoint import load_model_checkpoint
from src.models.model_factory import create_model
from src.pipeline.model_artifacts import example_input_shape,export_config,pipeline_model_role
from src.pipeline.progress import pipeline_logger
from src.utils.progress import progress_bar


def _source_checkpoint(config:dict[str,Any],result_dir:Path) -> Path | None:
    reporting=config.get("reporting",{})
    declared=reporting.get("compression_source_result_dir")
    if declared:
        candidate=Path(str(declared))/"checkpoints"/"best.pt"
        return candidate if candidate.exists() else None
    method=str(reporting.get("compression_source_method") or "racmed")
    if method=="teacher":return None
    seed=int(config.get("seed",0))
    experiment_root=result_dir.parents[2]
    candidate=experiment_root/method/f"seed_{seed}"/"checkpoints"/"best.pt"
    return candidate if candidate.exists() else None


def _inputs(config:dict[str,Any],model:torch.nn.Module,role:str):
    shape=example_input_shape(config,role)
    image=torch.zeros(shape,dtype=torch.float32)
    if not bool(getattr(model,"accepts_prompts",False)):return image
    role_config={**config.get("model",{}),**config.get(role,{})}
    channels=int(getattr(model,"prompt_channels",role_config.get("prompt_channels",max(int(role_config.get("num_classes",2))-1,1))))
    return image,torch.zeros((shape[0],channels,*shape[2:]),dtype=torch.float32)


def _shape_signature(model:torch.nn.Module,values) -> tuple:
    tensors=values if isinstance(values,tuple) else (values,)
    return (
        type(model).__module__,type(model).__qualname__,
        tuple(tuple(value.shape) for value in tensors),
        tuple((name,tuple(parameter.shape)) for name,parameter in model.named_parameters()),
    )


def backfill_compression_flops(results_dir:str|Path,force:bool=False) -> list[dict[str,Any]]:
    """Populate dense FLOPs for every compression result below an experiment root."""

    root=Path(results_dir);paths=sorted(root.glob("compressed/*/seed_*/compression/compression_metrics.json"))
    if not paths:raise FileNotFoundError(f"No compression metrics found below {root}")
    cache:dict[tuple,float]={};rows=[]
    for metrics_path in progress_bar(paths,desc="FLOP profiling",total=len(paths),unit="artifact",leave=True):
        result_dir=metrics_path.parents[1]
        config_path=result_dir/"config.yaml"
        if not config_path.exists():raise FileNotFoundError(f"Missing persisted config: {config_path}")
        config=yaml.safe_load(config_path.read_text()) or {}
        metrics=json.loads(metrics_path.read_text())
        if not force and metrics.get("baseline_flops") is not None and metrics.get("flops") is not None:
            rows.append({"result_dir":str(result_dir),**metrics});continue
        role=pipeline_model_role(config,"compression");model_config=export_config(config,role)
        source=create_model(model_config,role)
        source_checkpoint=_source_checkpoint(config,result_dir)
        if source_checkpoint is not None:source,_=load_model_checkpoint(source_checkpoint,source)
        pruned=create_model(model_config,role)
        pruned,_=load_model_checkpoint(result_dir/"compression"/"pruned.pt",pruned)
        source=source.cpu().eval();pruned=pruned.cpu().eval()
        source_inputs=_inputs(model_config,source,role);pruned_inputs=_inputs(model_config,pruned,role)
        def profile(model,values):
            signature=_shape_signature(model,values)
            if signature not in cache:cache[signature]=float(estimate_flops(model,values,strict=True))
            return cache[signature]
        baseline=profile(source,source_inputs);compressed=profile(pruned,pruned_inputs)
        metrics.update({
            "baseline_flops":baseline,"flops":compressed,
            "baseline_gflops":baseline/1e9,"gflops":compressed/1e9,
            "flops_reduction":1-compressed/baseline if baseline else None,
            "flops_definition":"dense multiply-add operations reported by fvcore",
        })
        metrics_path.write_text(json.dumps(metrics,indent=2))
        pipeline_logger().info(
            "FLOP_PROFILE | result=%s baseline_gflops=%.4f compressed_gflops=%.4f reduction=%.4f",
            result_dir,baseline/1e9,compressed/1e9,float(metrics["flops_reduction"] or 0.0),
        )
        rows.append({"result_dir":str(result_dir),**metrics})
        del source,pruned
    return rows
