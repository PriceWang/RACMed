"""Decision-oriented layer sensitivity profiling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .fake_quant import fake_quantize
from src.risk.risk_map import RiskMapBuilder
from src.utils.progress import progress_bar


def _logits(output):return output["logits"] if isinstance(output,dict) else output


_RISK_COMPONENTS = (
    "feature_drift",
    "output_drop",
    "gradient_sensitivity",
    "quantization_error",
    "high_risk_activation_correlation",
)


def aggregate_layer_risk(raw:dict[str,dict[str,float]],betas:tuple[float,...]) -> dict[str,dict[str,float]]:
    """Normalize each risk factor across layers before weighted aggregation."""

    if not raw:
        return raw
    for component in _RISK_COMPONENTS:
        values=[float(row[component]) for row in raw.values()]
        low,high=min(values),max(values)
        for row,value in zip(raw.values(),values):
            normalized=(value-low)/(high-low) if high>low else (0.5 if high>0 else 0.0)
            row[f"{component}_normalized"]=float(normalized)
    combined=[]
    for row in raw.values():
        combined.append(sum(float(beta)*row[f"{component}_normalized"] for beta,component in zip(betas,_RISK_COMPONENTS)))
    low,high=min(combined),max(combined)
    for row,value in zip(raw.values(),combined):
        row["risk"]=(value-low)/(high-low) if high>low else 0.5
    return raw


@torch.inference_mode()
def build_profile_risk_map(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    risk_config: dict | None = None,
    forward_kwargs: dict | None = None,
) -> torch.Tensor:
    """Build the spatial risk map used for activation-correlation profiling.

    A model profiled against itself has zero disagreement, while boundary,
    small-component, class-rarity, and predictive-entropy risks remain active.
    This is the appropriate target-aware signal when compressing a pretrained
    teacher directly.
    """

    if task_type != "classification" and not str(task_type).startswith("segmentation"):
        raise ValueError(f"Unsupported compression risk task: {task_type}")
    device = next(model.parameters()).device
    asynchronous=device.type=="cuda"
    image = inputs.to(device,non_blocking=asynchronous)
    target = targets.to(device,non_blocking=asynchronous)
    kwargs = {
        key: value.to(device,non_blocking=asynchronous) if isinstance(value, torch.Tensor) else value
        for key, value in (forward_kwargs or {}).items()
    }
    model.eval()
    logits = _logits(model(image, **kwargs))
    options = {
        key: value
        for key, value in (risk_config or {}).items()
        if hasattr(RiskMapBuilder, key)
    }
    if isinstance(options.get("teacher_temperature"),str):
        if options["teacher_temperature"].lower()!="auto":
            raise ValueError(
                "Compression profiling accepts numeric teacher_temperature or 'auto'."
            )
        # Validation fitting belongs to training. Layer profiling only needs
        # a stable within-batch risk ordering, so native frozen logits are used.
        options["teacher_temperature"]=1.0
    builder = RiskMapBuilder(task_type=task_type, **options)
    return builder(logits, logits, target, image)["risk_map"].detach()


class LayerRiskProfiler:
    """Estimate feature drift, output drop, gradients, quantization error and risk correlation."""
    def __init__(
        self,
        beta_feat=1.,
        beta_out=1.,
        beta_grad=1.,
        beta_quant=1.,
        beta_act=1.,
        perturbation=.05,
        perturbation_mode="noise",
        channel_ablation_ratio=.1,
    ):
        self.betas=(beta_feat,beta_out,beta_grad,beta_quant,beta_act)
        self.perturbation=float(perturbation)
        self.perturbation_mode=str(perturbation_mode)
        self.channel_ablation_ratio=float(channel_ablation_ratio)
        if self.perturbation_mode not in {"noise","channel_ablation"}:
            raise ValueError(
                "Layer-risk perturbation_mode must be 'noise' or 'channel_ablation'."
            )
        if not 0 < self.channel_ablation_ratio <= 1:
            raise ValueError("channel_ablation_ratio must be in (0, 1].")

    def profile(self,model:torch.nn.Module,inputs:torch.Tensor,targets:torch.Tensor|None=None,risk_map:torch.Tensor|None=None,loss_fn=None,metric_fn=None,max_layers:int|None=None,forward_kwargs:dict|None=None):
        device=next(model.parameters()).device;asynchronous=device.type=="cuda";inputs=inputs.to(device,non_blocking=asynchronous);model.eval();modules={n:m for n,m in model.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear))}
        kwargs={key:(value.to(device,non_blocking=asynchronous) if isinstance(value,torch.Tensor) else value) for key,value in (forward_kwargs or {}).items()}
        def forward():return _logits(model(inputs,**kwargs))
        if max_layers is not None and int(max_layers) > 0:
            modules=dict(list(modules.items())[:int(max_layers)])
        activations={};handles=[m.register_forward_hook(lambda mod,inp,out,n=n:activations.__setitem__(n,out.detach())) for n,m in modules.items()]
        with torch.no_grad():baseline=forward();base_acts={k:v.clone() for k,v in activations.items()};base_metric=float(metric_fn(baseline,targets.to(device))) if metric_fn and targets is not None else None
        gradient={n:0. for n in modules}
        if targets is not None and loss_fn is not None:
            model.zero_grad();loss_fn(forward(),targets.to(device)).backward()
            gradient={n:float(m.weight.grad.norm()/max(m.weight.numel()**0.5,1.0)) if m.weight.grad is not None else 0. for n,m in modules.items()};model.zero_grad(set_to_none=True);model.eval()
        raw={};total_layers=len(modules);progress_interval=max(1,total_layers//10) if total_layers else 1
        layers=progress_bar(modules.items(),desc="Layer-risk profiling",total=total_layers,unit="layer",leave=True)
        for layer_index,(name,module) in enumerate(layers,start=1):
            original=module.weight.detach().clone()
            original_bias=module.bias.detach().clone() if module.bias is not None else None
            with torch.no_grad():
                if self.perturbation_mode=="channel_ablation":
                    importance=original.flatten(1).abs().mean(1)
                    count=max(1,int(round(importance.numel()*self.channel_ablation_ratio)))
                    indices=importance.argsort()[:min(count,importance.numel())]
                    module.weight[indices]=0
                    if module.bias is not None:module.bias[indices]=0
                else:
                    module.weight.add_(
                        torch.randn_like(module.weight)
                        * original.std().clamp_min(1e-8)
                        * self.perturbation
                    )
                perturbed=forward()
                module.weight.copy_(original)
                if original_bias is not None:module.bias.copy_(original_bias)
            changed=activations.get(name,base_acts[name]);base=base_acts[name]
            feat=float((1-F.cosine_similarity(base.flatten(1),changed.flatten(1),dim=1)).mean().clamp_min(0))
            out=max(0.,base_metric-float(metric_fn(perturbed,targets.to(device)))) if base_metric is not None else float(F.mse_loss(perturbed,baseline)/(baseline.pow(2).mean()+1e-8));quant=float(F.mse_loss(fake_quantize(base),base)/(base.pow(2).mean()+1e-8))
            corr=0.
            if risk_map is not None:
                if risk_map.ndim == 1:
                    # Classification risk is sample-level.  Reduce each layer
                    # activation to one magnitude per sample before measuring
                    # whether the layer preferentially supports risky cases.
                    r=risk_map.to(device).flatten()
                    a=base.abs().flatten(1).mean(1)
                elif base.ndim>=4:
                    mode="bilinear" if base.ndim==4 else "trilinear"
                    r=F.interpolate(
                        risk_map.to(device).unsqueeze(1),
                        size=base.shape[2:],mode=mode,align_corners=False,
                    )[:,0].flatten()
                    a=base.abs().mean(1).flatten()
                else:
                    r=a=None
                if r is not None and r.numel()>1 and a.numel()==r.numel():
                    centered_a=a-a.mean();centered_r=r-r.mean()
                    denominator=(centered_a.square().mean().sqrt()*centered_r.square().mean().sqrt()).clamp_min(1e-8)
                    corr=max(float((centered_a*centered_r).mean()/denominator),0.)
            raw[name] = {"feature_drift":feat,"output_drop":out,"gradient_sensitivity":gradient[name],"quantization_error":quant,"high_risk_activation_correlation":corr}
            layers.set_postfix(layer=name,risk_signal=f"{feat+out+quant+corr:.3g}")
            if layer_index%progress_interval==0 or layer_index==total_layers:
                from src.pipeline.progress import pipeline_logger
                pipeline_logger().info("LAYER_RISK_PROGRESS | layer=%d/%d name=%s",layer_index,total_layers,name)
        for handle in handles:handle.remove()
        return aggregate_layer_risk(raw,self.betas)

    @staticmethod
    def save(scores,path):
        import pandas as pd
        pd.DataFrame.from_dict(scores,orient="index").rename_axis("layer").reset_index().to_csv(path,index=False)
