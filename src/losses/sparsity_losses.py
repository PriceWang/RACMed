"""Gate and batch-normalization sparsity objectives."""

import torch


def sparsity_loss(model: torch.nn.Module | None) -> torch.Tensor:
    if model is None:
        return torch.tensor(0.0)
    values = []
    for module in model.modules():
        if hasattr(module, "gate") and isinstance(module.gate, torch.Tensor):
            values.append(module.gate.abs().mean())
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) and module.affine:
            values.append(module.weight.abs().mean())
    return torch.stack(values).mean() if values else next(model.parameters()).sum() * 0
