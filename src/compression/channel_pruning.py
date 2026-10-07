"""Structured magnitude and mask-based channel pruning with optional structural backend."""

from __future__ import annotations
import logging
import torch


class MaskBasedPruner:
    def __init__(self,model):self.model=model
    def apply(self,policy):
        modules=dict(self.model.named_modules())
        with torch.no_grad():
            for name,ratio in policy.items():
                module=modules.get(name)
                if not isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):continue
                count=int(module.weight.shape[0]*float(ratio))
                if count<=0:continue
                importance=module.weight.flatten(1).abs().mean(1);indices=importance.argsort()[:count];module.weight[indices]=0
                if module.bias is not None:module.bias[indices]=0
        return self.model
    def effective_parameters(self):return sum(int(torch.count_nonzero(p)) for p in self.model.parameters())


class ElasticPruner(MaskBasedPruner):
    """Apply safe layer-wise magnitude masks using risk-derived sparsity ratios.

    The fallback intentionally avoids zeroing whole output channels because
    channel dependencies through normalization and skip connections are not
    structurally rewritten without a dependency-graph backend.
    """

    def apply(self, policy):
        modules = dict(self.model.named_modules())
        with torch.no_grad():
            for name, ratio in policy.items():
                module = modules.get(name)
                if not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                    continue
                flat = module.weight.view(-1)
                count = int(flat.numel() * float(ratio))
                if count <= 0:
                    continue
                indices = flat.abs().argsort()[:count]
                flat[indices] = 0
        return self.model


def magnitude_prune(model,ratio:float,chunk_size:int=1_000_000):
    """Global unstructured magnitude baseline with bounded working memory."""

    parameters=[parameter for parameter in model.parameters() if parameter.ndim>1]
    total=sum(parameter.numel() for parameter in parameters)
    prune_count=int(total*min(max(float(ratio),0.),1.))
    if prune_count<=0:return model
    smallest=None
    with torch.no_grad():
        for parameter in parameters:
            values=parameter.detach().abs().flatten()
            for chunk in values.split(chunk_size):
                candidates=chunk if smallest is None else torch.cat((smallest,chunk))
                keep=min(prune_count,candidates.numel())
                smallest=torch.topk(candidates,keep,largest=False,sorted=False).values if candidates.numel()>keep else candidates
        threshold=smallest.max()
        below=sum(int(torch.count_nonzero(parameter.detach().abs()<threshold)) for parameter in parameters)
        ties_remaining=max(0,prune_count-below)
        for parameter in parameters:
            flat=parameter.view(-1);magnitudes=flat.abs();mask=magnitudes<threshold
            if ties_remaining:
                ties=torch.nonzero(magnitudes==threshold,as_tuple=False).flatten()[:ties_remaining]
                mask[ties]=True;ties_remaining-=ties.numel()
            flat[mask]=0
    return model


def optional_structural_prune(
    model,
    example_inputs,
    policy,
    *,
    forward_fn=None,
    ignored_layers=None,
):
    """Rewrite channels with ``torch-pruning`` or fall back to safe masks.

    ``policy`` maps module names to output-channel pruning ratios. The optional
    ``forward_fn`` supports models whose public forward requires prompts or
    returns dictionaries. Prediction heads should be passed through
    ``ignored_layers`` so their output semantics remain unchanged.
    """

    try:
        import torch_pruning as tp
    except ImportError:
        logging.warning("torch-pruning unavailable; using mask-based pruning fallback.")
        return MaskBasedPruner(model).apply(policy)
    modules=dict(model.named_modules())
    ratio_dict={
        modules[name]:min(max(float(ratio),0.0),0.95)
        for name,ratio in policy.items()
        if name in modules
        and float(ratio)>0
        and isinstance(modules[name],(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear))
    }
    if not ratio_dict:
        return model
    try:
        pruner=tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            tp.importance.MagnitudeImportance(p=2),
            pruning_ratio_dict=ratio_dict,
            ignored_layers=list(ignored_layers or []),
            iterative_steps=1,
            forward_fn=forward_fn,
        )
        pruner.step()
        return model
    except Exception as exc:
        logging.warning("Structural pruning failed (%s); using mask-based pruning fallback.",exc)
        return MaskBasedPruner(model).apply(policy)
