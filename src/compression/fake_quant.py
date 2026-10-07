"""Symmetric per-tensor fake quantization."""

import torch


def fake_quantize(tensor:torch.Tensor,bits:int=8)->torch.Tensor:
    qmax=2**(bits-1)-1;scale=tensor.detach().abs().max().clamp_min(1e-8)/qmax
    return (tensor/scale).round().clamp(-qmax,qmax)*scale
