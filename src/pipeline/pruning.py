"""Layer-risk profiling and pruning stage."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch

from src.compression.channel_pruning import (
    ElasticPruner,
    magnitude_prune,
    optional_structural_prune,
)
from src.compression.compression_metrics import compression_metrics,estimate_flops
from src.compression.elastic_policy import (
    ElasticCompressionPolicy,
    calibrate_global_pruning_policy,
    policy_prunable_sparsity,
    risk_adjusted_magnitude_policy,
)
from src.compression.layer_risk import (
    LayerRiskProfiler,
    build_profile_risk_map,
)
from src.compression.profile_objectives import build_profile_objectives
from src.datasets.dataset_factory import create_dataloaders
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.engine.trainer import Trainer
from src.models.model_factory import create_model
from src.pipeline.model_artifacts import export_config, pipeline_checkpoint, pipeline_model_role
from src.utils.device import get_device
from src.utils.seed import seed_everything
from src.pipeline.progress import pipeline_logger


_PRUNABLE=(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)


def _post_prune_finetune(
    model:torch.nn.Module,
    config:dict[str,Any],
    settings:dict[str,Any],
) -> torch.nn.Module:
    """Recover a structurally pruned model with validation-only model selection."""

    if not settings or not bool(settings.get("enabled",False)):
        return model
    cfg=copy.deepcopy(config)
    training={**cfg.get("training",{}),**{key:value for key,value in settings.items() if key!="enabled"}}
    training["evaluate_initial_checkpoint"]=True
    training["resume"]=None
    cfg["training"]=training
    mode=str(settings.get("mode",cfg.get("reporting",{}).get("compression_source_method","racmed")))
    if mode not in {"racmed","compressed"}:mode="racmed"
    trainer=Trainer(cfg,mode)
    trainer.model=model.to(trainer.device)
    lr=float(training.get("lr",1e-4));weight_decay=float(training.get("weight_decay",1e-5))
    optimizer_name=str(training.get("optimizer","adamw")).lower()
    if optimizer_name=="sgd":
        trainer.optimizer=torch.optim.SGD(trainer.model.parameters(),lr=lr,momentum=.9,weight_decay=weight_decay)
    elif optimizer_name=="adam":
        trainer.optimizer=torch.optim.Adam(trainer.model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=lr,weight_decay=weight_decay)
    trainer.scheduler=(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,T_max=max(int(training.get("epochs",1)),1)
        )
        if training.get("scheduler","cosine")=="cosine" else None
    )
    pipeline_logger().info(
        "POST_PRUNE_FINETUNE | mode=%s epochs=%d lr=%.3g validation_selection=true",
        mode,int(training.get("epochs",1)),lr,
    )
    trainer.fit()
    return trainer.model


def calibrated_structural_prune(
    model:torch.nn.Module,
    example_inputs,
    pruning:dict[str,float],
    target_parameter_reduction:float,
    *,
    forward_fn=None,
    protected_layer_patterns=("head","classifier",".out."),
    tolerance:float=.01,
    max_attempts:int=5,
) -> tuple[torch.nn.Module,dict[str,float],float]:
    """Scale a layer allocation until physical parameter reduction matches target."""

    target=max(0.,min(float(target_parameter_reduction),.95))
    original=copy.deepcopy(model)
    baseline=sum(parameter.numel() for parameter in original.parameters())
    scale=1.;best_model=None;best_policy=None;best_reduction=-1.;best_error=float("inf")
    for attempt in range(max(int(max_attempts),1)):
        candidate=copy.deepcopy(original)
        candidate_policy={name:min(max(float(value)*scale,0.),.95) for name,value in pruning.items()}
        ignored=[
            module for name,module in candidate.named_modules()
            if any(pattern in name for pattern in protected_layer_patterns)
        ]
        candidate=optional_structural_prune(
            candidate,example_inputs,candidate_policy,forward_fn=forward_fn,ignored_layers=ignored
        )
        reduction=1-sum(parameter.numel() for parameter in candidate.parameters())/max(baseline,1)
        error=abs(reduction-target)
        pipeline_logger().info(
            "STRUCTURAL_BUDGET | attempt=%d scale=%.4f target=%.4f actual=%.4f",
            attempt+1,scale,target,reduction,
        )
        if error<best_error:
            best_model,best_policy,best_reduction,best_error=candidate,candidate_policy,reduction,error
        if error<=tolerance:
            break
        if reduction<=0:
            raise RuntimeError("Structural pruning did not reduce physical parameters; a dependency-graph backend is required.")
        scale=max(.01,min(scale*target/reduction,10.))
    if best_model is None or best_policy is None:
        raise RuntimeError("Structural pruning budget calibration produced no candidate model.")
    if best_error>max(tolerance,.025):
        raise RuntimeError(
            f"Unable to match physical pruning target {target:.3f}; best reduction was {best_reduction:.3f}."
        )
    return best_model,best_policy,float(best_reduction)


def _bounded_profile_batch(batch:dict[str,Any],spatial_size:list[int]|tuple[int,...]|None) -> dict[str,Any]:
    """Crop a deterministic foreground-centered profiling region to bound 3D memory."""

    if spatial_size is None:return batch
    image=batch["image"];target=batch["target"]
    requested=tuple(int(value) for value in spatial_size)
    spatial=tuple(int(value) for value in image.shape[2:])
    if len(requested)!=len(spatial):raise ValueError(f"Profile crop {requested} does not match input spatial shape {spatial}")
    crop=tuple(min(current,limit) for current,limit in zip(spatial,requested))
    if crop==spatial:return batch
    prefix=target.ndim-len(spatial)
    foreground=target.bool()
    if prefix>0:foreground=foreground.any(dim=tuple(range(prefix)))
    center=torch.tensor([value//2 for value in spatial])
    if foreground.any():center=foreground.nonzero().float().mean(0).round().long()
    starts=[max(0,min(int(position)-length//2,current-length)) for position,current,length in zip(center,spatial,crop)]
    slices=tuple(slice(start,start+length) for start,length in zip(starts,crop))
    bounded=dict(batch)
    bounded["image"]=image[(slice(None),slice(None),*slices)]
    bounded["target"]=target[(... ,*slices)]
    pipeline_logger().info("PROFILE_CROP | original=%s crop=%s",spatial,crop)
    return bounded


def run_pruning(config: dict[str, Any], checkpoint: str | Path | None = None, debug: bool = False) -> dict[str, Any]:
    """Profile layer risk, apply the selected policy, and save the pruned model."""

    cfg = config
    seed_everything(int(cfg.get("seed", 0)))
    out = Path(cfg.get("output_dir", "outputs/experiment"))
    comp = out / "compression"
    comp.mkdir(parents=True, exist_ok=True)
    role = pipeline_model_role(cfg, "compression")
    model_cfg = export_config(cfg, role)
    device = get_device(cfg.get("device", "auto"))
    gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    model = create_model(model_cfg, role).to(device)
    selected = pipeline_checkpoint(cfg, out, "compression", checkpoint)
    if selected is not None:
        if not selected.exists():
            raise FileNotFoundError(f"Compression checkpoint not found: {selected}")
        load_checkpoint(selected, model)
    compression_cfg = cfg.get("compression", {})
    method=str(compression_cfg.get("pruning_method","elastic"))
    if method not in {"elastic","magnitude"}:raise ValueError(f"Unsupported pruning method: {method}")
    structural_backend=bool(compression_cfg.get("structural_backend",False))
    policy = ElasticCompressionPolicy(
        float(compression_cfg.get("max_prune_ratio", 0.5)),
        float(compression_cfg.get("high_threshold", 0.7)),
        tuple(compression_cfg.get("protected_layer_patterns", ("head", "classifier", ".out."))),
        compression_cfg.get("max_int8_fraction"),
    )
    target_sparsity = compression_cfg.get("target_global_sparsity")
    ratio = float(target_sparsity) if target_sparsity is not None else float(compression_cfg.get("max_prune_ratio", 0.5))
    scores={};profile_batch=None
    if method=="elastic":
        betas=tuple(
            float(compression_cfg.get(key,1.0))
            for key in ("beta_feat","beta_out","beta_grad","beta_quant","beta_act")
        )
        split = compression_cfg.get("profile_split") or ("train" if role != "compressed_model" else "val")
        batch = next(iter(create_dataloaders(model_cfg, debug)[split]));profile_batch=batch
        profile_size=compression_cfg.get("profile_crop_size")
        if profile_size is None and str(cfg.get("task_type",""))=="segmentation3d":profile_size=cfg.get("dataset",{}).get("crop_size") or cfg.get("deployment",{}).get("input_size")
        batch=_bounded_profile_batch(batch,profile_size)
        forward_kwargs = {key: batch[key] for key in ("boxes", "box_valid") if key in batch} or None
        risk_map = None
        if compression_cfg.get("use_risk_map", True):
            risk_map = build_profile_risk_map(model,batch["image"],batch["target"],str(cfg["task_type"]),cfg.get("risk", {}),forward_kwargs)
        classes = int(cfg.get("model", {}).get("num_classes", cfg.get("dataset", {}).get("num_classes", 2)))
        loss_fn, metric_fn = build_profile_objectives(str(cfg.get("task_type", "classification")), classes, cfg.get("loss", {}))
        profiler = LayerRiskProfiler(
            beta_feat=betas[0],beta_out=betas[1],beta_grad=betas[2],beta_quant=betas[3],beta_act=betas[4],
            perturbation=float(compression_cfg.get("profile_perturbation",.05)),
            perturbation_mode=str(compression_cfg.get("profile_perturbation_mode","noise")),
            channel_ablation_ratio=float(
                compression_cfg.get(
                    "profile_channel_ablation_ratio",
                    min(float(target_sparsity),.1) if target_sparsity is not None else .1,
                )
            ),
        )
        scores = profiler.profile(model,batch["image"],targets=batch["target"],risk_map=risk_map,loss_fn=loss_fn,metric_fn=metric_fn,max_layers=compression_cfg.get("max_profile_layers"),forward_kwargs=forward_kwargs)
        pipeline_logger().info("LAYER_RISK | role=%s profiled_layers=%d method=%s",role,len(scores),method)
        LayerRiskProfiler.save(scores,comp/"layer_risk_scores.csv")
        pruning,precision=policy.build(scores)
        if target_sparsity is not None:
            if compression_cfg.get("allocation_mode") == "risk_adjusted_magnitude":
                pruning = risk_adjusted_magnitude_policy(model,scores,float(target_sparsity),float(compression_cfg.get("risk_protection_strength", 2.0)),tuple(compression_cfg.get("protected_layer_patterns", ("head", "classifier", ".out."))))
            else:
                pruning = calibrate_global_pruning_policy(
                    model, pruning, float(target_sparsity), float(compression_cfg.get("max_layer_prune_ratio", 0.95))
                )
    else:
        pruning={name:ratio for name,module in model.named_modules() if isinstance(module,_PRUNABLE)}
        precision_value="int8" if str(compression_cfg.get("quantization","")).lower() in {"int8","uniform_int8"} else "fp32"
        precision={name:precision_value for name in pruning}
        pipeline_logger().info(
            "LAYER_RISK | skipped method=%s reason=static_policy layers=%d",
            method,
            len(pruning),
        )
    baseline = sum(parameter.numel() for parameter in model.parameters());baseline_flops=None;example_inputs=None
    achieved_policy_sparsity=policy_prunable_sparsity(model,pruning)
    calibrated_physical_reduction=None
    if structural_backend and method in {"elastic","magnitude"}:
        if profile_batch is None:
            split=compression_cfg.get("profile_split") or "val"
            profile_batch=next(iter(create_dataloaders(model_cfg,debug)[split]))
        image=profile_batch["image"].to(device)
        example_inputs=image;forward_fn=None
        if getattr(model,"accepts_prompts",False):
            boxes=profile_batch["boxes"].to(device);valid=profile_batch.get("box_valid")
            valid=valid.to(device) if valid is not None else None
            prompt_maps=model._prompt_maps(image,boxes,valid)
            example_inputs=(image,prompt_maps)
            forward_fn=lambda current,inputs:current.forward_with_prompt_maps(*inputs,return_features=False)["logits"]
        baseline_flops=estimate_flops(model,example_inputs)
        pruned_model,pruning,calibrated_physical_reduction=calibrated_structural_prune(
            model,
            example_inputs,
            pruning,
            float(compression_cfg.get("target_physical_parameter_reduction",target_sparsity if target_sparsity is not None else ratio)),
            forward_fn=forward_fn,
            protected_layer_patterns=tuple(compression_cfg.get("protected_layer_patterns",("head","classifier",".out."))),
            tolerance=float(compression_cfg.get("structural_budget_tolerance",.01)),
        )
        achieved_policy_sparsity=policy_prunable_sparsity(model,pruning)
        model=pruned_model
    elif method == "magnitude":
        magnitude_prune(model, ratio)
    else:
        ElasticPruner(model).apply(pruning)
    model=_post_prune_finetune(model,cfg,dict(compression_cfg.get("post_prune_finetune",{})))
    policy.save(pruning, precision, comp)
    artifact = comp / "pruned.pt"
    save_checkpoint(artifact, model, serialize_model=structural_backend)
    metrics = compression_metrics(model, baseline, artifact)
    compressed_flops=estimate_flops(model,example_inputs) if example_inputs is not None else None
    metrics.update({
        "target_prunable_sparsity": target_sparsity,
        "policy_prunable_sparsity": achieved_policy_sparsity,
        "allocation_mode": "global_magnitude" if method == "magnitude" else compression_cfg.get("allocation_mode", "layer_ratio"),
        "pruning_method": method,
        "layer_risk_profiled":method=="elastic",
        "structural_backend":structural_backend,
        "physical_parameter_reduction":1-sum(parameter.numel() for parameter in model.parameters())/max(baseline,1),
        "target_physical_parameter_reduction":float(compression_cfg.get("target_physical_parameter_reduction",target_sparsity if target_sparsity is not None else ratio)) if structural_backend else None,
        "calibrated_physical_parameter_reduction":calibrated_physical_reduction,
        "baseline_flops":baseline_flops,
        "flops":compressed_flops,
        "flops_reduction":1-compressed_flops/baseline_flops if baseline_flops and compressed_flops is not None else None,
        "source_checkpoint_size_mb":selected.stat().st_size/1024**2 if selected is not None and selected.exists() else None,
        "artifact_checkpoint_size_mb":artifact.stat().st_size/1024**2,
    })
    (comp / "compression_metrics.json").write_text(json.dumps(metrics, indent=2))
    pipeline_logger().info(
        "PRUNING_RESULT | method=%s target_sparsity=%s effective_params=%s compression_ratio=%.3f",
        method,
        target_sparsity,
        metrics.get("effective_params"),
        float(metrics.get("compression_ratio", 1.0)),
    )
    return {"checkpoint": str(artifact), "metrics": metrics}
