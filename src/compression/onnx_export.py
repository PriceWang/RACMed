"""ONNX export for dictionary-returning RACMed models."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import torch,shutil,subprocess,warnings


class LogitsAdapter(torch.nn.Module):
    """Expose tensor logits for deployment formats that cannot return dictionaries."""

    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,prompt_maps=None):
        out=self.model.forward_with_prompt_maps(x,prompt_maps) if prompt_maps is not None else self.model(x)
        return out["logits"] if isinstance(out,dict) else out


def _as_inputs(example_input):
    return tuple(example_input) if isinstance(example_input,(tuple,list)) else (example_input,)


@contextmanager
def _validated_fixed_resize(model,example_input,three_dimensional_mode="native"):
    """Use constant-scale decoder resize only when it is numerically exact.

    Compact U-Nets use a skip tensor's runtime shape during ordinary training
    and evaluation. Deployment inputs are fixed and divisible by every pooling
    stage, so the same resize can be represented as a constant 2x operation.
    Removing Shape/Gather/Concat shape subgraphs avoids fragile TensorRT EP and
    Core ML lowering. The original mode is restored even if export fails.
    """

    from src.models.segmentation2d import CompactUNet2D

    modules=[module for module in model.modules() if isinstance(module,CompactUNet2D)]
    previous=[getattr(module,"_fixed_deployment_upsample",False) for module in modules]
    adapter=LogitsAdapter(model).eval()
    inputs=_as_inputs(example_input)
    if not modules:
        yield adapter,inputs
        return
    with torch.inference_mode():
        reference=adapter(*inputs)
    try:
        resize_mode="separable" if three_dimensional_mode=="separable" else "native"
        for module in modules:module._fixed_deployment_upsample=resize_mode
        with torch.inference_mode():
            deployment_output=adapter(*inputs)
        if reference.shape!=deployment_output.shape:
            raise RuntimeError(
                "Fixed-resize deployment export changed output shape: "
                f"{tuple(reference.shape)} != {tuple(deployment_output.shape)}"
            )
        torch.testing.assert_close(reference,deployment_output,rtol=1e-5,atol=1e-6)
        yield adapter,inputs
    finally:
        for module,value in zip(modules,previous):module._fixed_deployment_upsample=value


def export_onnx(model,example_input,path,dynamic_axes=True,opset=17):
    try:import onnx  # noqa: F401
    except ImportError as exc:raise ImportError("Install onnx to export models") from exc
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);model.eval()
    multiple=isinstance(example_input,(tuple,list));input_names=["input","prompt_maps"] if multiple else ["input"]
    axes={name:{0:"batch"} for name in [*input_names,"logits"]} if dynamic_axes else None
    with _validated_fixed_resize(model,example_input,three_dimensional_mode="native") as (adapter,inputs):
        trace_inputs=inputs if len(inputs)>1 else inputs[0]
        torch.onnx.export(adapter,trace_inputs,path,input_names=input_names,output_names=["logits"],dynamic_axes=axes,opset_version=opset,dynamo=False)
    return path


def export_coreml(model, example_input, path):
    """Export a numerically checked, traced ML Program for Apple Silicon.

    Compact U-Nets normally resize decoder features to a dynamic skip-tensor
    shape. Recent TorchScript graphs express that shape through ``aten::Int``
    nodes that Core ML Tools cannot lower reliably. For fixed deployment input
    sizes, a 2x resize is equivalent. The switch is temporary and conversion is
    allowed only after comparing the original and export-mode outputs.
    """

    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError("Install coremltools on macOS to export CoreML models") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_model = model.cpu().eval()
    inputs = _as_inputs(example_input)
    inputs = tuple(tensor.cpu() for tensor in inputs)
    trace_inputs = inputs if len(inputs) > 1 else inputs[0]

    with _validated_fixed_resize(source_model,inputs,three_dimensional_mode="separable") as (adapter,_):
        traced = torch.jit.trace(adapter, trace_inputs)
        names = ("input", "prompt_maps")[: len(inputs)]
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name=name, shape=tuple(tensor.shape)) for name, tensor in zip(names, inputs)],
        )
        converted.save(str(path))
    return path


def export_tensorrt_hook(onnx_path,engine_path,fp16=True):
    """Build a TensorRT engine with `trtexec`, or return None on unsupported hosts."""
    executable=shutil.which("trtexec")
    if executable is None:warnings.warn("TensorRT trtexec unavailable; ONNX artifact remains deployable.");return None
    command=[executable,f"--onnx={onnx_path}",f"--saveEngine={engine_path}"]+(["--fp16"] if fp16 else [])
    subprocess.run(command,check=True);return Path(engine_path)
