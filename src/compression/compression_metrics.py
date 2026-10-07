"""Model size and compression ratio metrics."""

import copy
import logging
from pathlib import Path
import torch


def estimate_flops(model:torch.nn.Module,example_inputs,strict:bool=False) -> float | None:
    """Estimate dense operation count with fvcore when it supports the graph."""

    try:
        from fvcore.nn import FlopCountAnalysis

        class _Logits(torch.nn.Module):
            def __init__(self,current):super().__init__();self.current=current
            def forward(self,*inputs):
                output=self.current.forward_with_prompt_maps(*inputs,return_features=False) if len(inputs)>1 else self.current(inputs[0],return_features=False)
                return output["logits"] if isinstance(output,dict) else output

        # TorchScript/FVCore may register temporary tracing hooks on every module.
        # Profile an isolated copy so a subsequent structural checkpoint can be
        # serialized even when profiling raises or a backend leaves hooks behind.
        profiled_model=copy.deepcopy(model).to("cpu").eval()
        values=tuple(example_inputs) if isinstance(example_inputs,(tuple,list)) else (example_inputs,)
        values=tuple(value.detach().cpu() if torch.is_tensor(value) else value for value in values)
        return float(FlopCountAnalysis(_Logits(profiled_model),values).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total())
    except Exception as exc:
        logging.getLogger("racmed.pipeline").warning(
            "FLOP_PROFILE_FAIL | model=%s error=%s",type(model).__name__,exc,
        )
        if strict:raise
        return None


def compression_metrics(model,baseline_params=None,model_path=None):
    params=sum(p.numel() for p in model.parameters());effective=sum(int(torch.count_nonzero(p)) for p in model.parameters())
    prunable=[module.weight for module in model.modules() if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear))]
    prunable_params=sum(weight.numel() for weight in prunable);effective_prunable=sum(int(torch.count_nonzero(weight)) for weight in prunable)
    size=Path(model_path).stat().st_size/1024**2 if model_path and Path(model_path).exists() else sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    return {"params":params,"effective_params":effective,"prunable_params":prunable_params,"effective_prunable_params":effective_prunable,
            "prunable_sparsity":1-effective_prunable/max(prunable_params,1),"model_size_mb":size,
            "compression_ratio":(baseline_params/effective if baseline_params else params/max(effective,1))}
