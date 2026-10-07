"""Convert layer risk into pruning and bit-width decisions."""

from __future__ import annotations
import json
from pathlib import Path

import torch


_PRUNABLE = (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)


def policy_prunable_sparsity(model:torch.nn.Module,policy:dict[str,float]) -> float:
    """Return the policy's achieved sparsity over all Conv/Linear weights."""

    modules=dict(model.named_modules());total=0;removed=0
    for name,module in modules.items():
        if not isinstance(module,_PRUNABLE):continue
        count=module.weight.numel();total+=count
        removed+=int(count*max(0.0,min(float(policy.get(name,0.0)),1.0)))
    return removed/max(total,1)


def calibrate_global_pruning_policy(
    model:torch.nn.Module,
    policy:dict[str,float],
    target_sparsity:float,
    max_layer_ratio:float=0.95,
    tolerance:float=1e-4,
) -> dict[str,float]:
    """Rescale layer ratios to a matched global prunable-weight budget."""

    target=max(0.0,min(float(target_sparsity),1.0));maximum=max(0.0,min(float(max_layer_ratio),1.0))
    if target==0:return {name:0.0 for name in policy}
    if not any(float(value)>0 for value in policy.values()):raise ValueError("Cannot calibrate a zero pruning policy to non-zero sparsity.")
    def scaled(scale:float)->dict[str,float]:return {name:min(maximum,max(0.0,float(value)*scale)) for name,value in policy.items()}
    high=1.0
    while policy_prunable_sparsity(model,scaled(high))<target and high<1e6:high*=2
    if policy_prunable_sparsity(model,scaled(high))+tolerance<target:
        raise ValueError(f"Target sparsity {target:.4f} is infeasible with max layer ratio {maximum:.4f}.")
    low=0.0;best=scaled(high)
    for _ in range(60):
        middle=(low+high)/2;candidate=scaled(middle);achieved=policy_prunable_sparsity(model,candidate)
        best=candidate
        if abs(achieved-target)<=tolerance:break
        if achieved<target:low=middle
        else:high=middle
    return best


def risk_adjusted_magnitude_policy(
    model:torch.nn.Module,
    scores:dict[str,dict[str,float]],
    target_sparsity:float,
    protection_strength:float=2.0,
    protected_layer_patterns=("head","classifier",".out."),
    chunk_size:int=1_000_000,
) -> dict[str,float]:
    """Allocate an exact global magnitude budget after layer-risk reweighting.

    Weight importance is multiplied by ``1 + strength * layer_risk``. Thus a
    zero-strength policy is the global magnitude baseline, while increasing
    strength progressively protects sensitive layers without abandoning global
    weight importance.
    """

    modules={name:module for name,module in model.named_modules() if isinstance(module,_PRUNABLE) and name in scores}
    patterns=tuple(protected_layer_patterns or ())
    protected=lambda name:any(pattern in name for pattern in patterns)
    total=sum(module.weight.numel() for module in modules.values())
    prune_count=int(total*max(0.0,min(float(target_sparsity),1.0)))
    available=sum(module.weight.numel() for name,module in modules.items() if not protected(name))
    if prune_count>available:raise ValueError("Target sparsity exceeds the weights available after prediction-head protection.")
    if prune_count<=0:return {name:0.0 for name in modules}
    smallest=None
    adjusted_by_layer={}
    with torch.no_grad():
        for name,module in modules.items():
            if protected(name):continue
            multiplier=1.0+float(protection_strength)*max(0.0,float(scores[name].get("risk",0.0)))
            adjusted=module.weight.detach().abs().flatten()*multiplier
            adjusted_by_layer[name]=adjusted
            for chunk in adjusted.split(chunk_size):
                candidates=chunk if smallest is None else torch.cat((smallest,chunk))
                keep=min(prune_count,candidates.numel())
                smallest=torch.topk(candidates,keep,largest=False,sorted=False).values if candidates.numel()>keep else candidates
        threshold=smallest.max()
        counts={name:(0 if protected(name) else int(torch.count_nonzero(adjusted_by_layer[name]<threshold))) for name in modules}
        remaining=prune_count-sum(counts.values())
        for name in modules:
            if protected(name) or remaining<=0:continue
            ties=int(torch.count_nonzero(adjusted_by_layer[name]==threshold))
            selected=min(remaining,ties);counts[name]+=selected;remaining-=selected
    return {
        name:(0.0 if counts[name]==0 else counts[name]/module.weight.numel()+1e-12)
        for name,module in modules.items()
    }


class ElasticCompressionPolicy:
    def __init__(self,max_prune_ratio=.5,high_threshold=.7,protected_layer_patterns=("head","classifier",".out."),max_int8_fraction=None):
        self.max_prune_ratio=max_prune_ratio;self.high_threshold=high_threshold;self.protected_layer_patterns=tuple(protected_layer_patterns or ())
        self.max_int8_fraction=None if max_int8_fraction is None else max(0.,min(float(max_int8_fraction),1.))
    def build(self,scores):
        protected=lambda name:any(pattern in name for pattern in self.protected_layer_patterns)
        pruning={name:(0. if protected(name) else max(0.,min(self.max_prune_ratio,self.max_prune_ratio*(1-values["risk"])))) for name,values in scores.items()}
        eligible=[
            name for name,values in sorted(scores.items(),key=lambda item:float(item[1]["risk"]))
            if not protected(name) and float(values["risk"])<=self.high_threshold
        ]
        if self.max_int8_fraction is not None:
            budget=int(len(scores)*self.max_int8_fraction)
            eligible=eligible[:budget]
        int8=set(eligible)
        precision={name:("int8" if name in int8 else "fp16") for name in scores}
        return pruning,precision
    @staticmethod
    def save(pruning,precision,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/"elastic_pruning_policy.json").write_text(json.dumps(pruning,indent=2));(directory/"pruning_policy.json").write_text(json.dumps(pruning,indent=2));(directory/"mixed_precision_policy.json").write_text(json.dumps(precision,indent=2))
