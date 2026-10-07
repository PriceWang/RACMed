"""ONNX Runtime dynamic and calibrated Q/DQ quantization entry points."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import logging
import numpy as np


class InMemoryCalibrationReader:
    """Rewindable ONNX Runtime calibration reader backed by input batches."""

    def __init__(self,records):
        self.records=list(records)
        if not self.records:raise ValueError("INT8 Q/DQ calibration requires at least one batch.")
        self.rewind()
    def get_next(self):return next(self._iterator,None)
    def rewind(self):self._iterator=iter(self.records)


def onnx_int8(input_path,output_path):
    try:
        from onnxruntime.quantization import quantize_dynamic,QuantType
    except ImportError as exc:raise ImportError("onnxruntime is required for ONNX INT8 quantization") from exc
    quantize_dynamic(str(input_path),str(output_path),weight_type=QuantType.QInt8);return Path(output_path)


def fold_int32_dequantize_constants(path: str | Path) -> int:
    """Fold INT32 bias DQ nodes into equivalent FP32 constants for TensorRT 10.3.

    ONNX Runtime's QDQ quantizer represents Conv/Gemm biases as INT32
    initializers followed by ``DequantizeLinear``. TensorRT 10.3 rejects that
    representation. Constant folding preserves the quantized graph's exact
    affine values while presenting an ordinary floating-point bias to the
    TensorRT parser.
    """

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError("onnx is required to make Q/DQ graphs TensorRT-compatible") from exc
    destination=Path(path)
    model=onnx.load(str(destination))
    graph=model.graph
    initializers={item.name:item for item in graph.initializer}
    consumer_counts={}
    for node in graph.node:
        for name in node.input:consumer_counts[name]=consumer_counts.get(name,0)+1
    replacements=[];removed_inputs=set();kept=[]
    for node in graph.node:
        quantized=initializers.get(node.input[0]) if node.op_type=="DequantizeLinear" and node.input else None
        if quantized is None or quantized.data_type!=onnx.TensorProto.INT32 or len(node.input)<2:
            kept.append(node);continue
        scale=initializers.get(node.input[1])
        zero_point=initializers.get(node.input[2]) if len(node.input)>2 and node.input[2] else None
        if scale is None or (zero_point is not None and zero_point.data_type!=onnx.TensorProto.INT32):
            kept.append(node);continue
        values=numpy_helper.to_array(quantized).astype(np.float32)
        scales=numpy_helper.to_array(scale).astype(np.float32)
        zeros=np.zeros_like(scales,dtype=np.float32) if zero_point is None else numpy_helper.to_array(zero_point).astype(np.float32)
        if scales.ndim==1 and values.ndim>1:
            axis=next((int(attribute.i) for attribute in node.attribute if attribute.name=="axis"),1)
            axis=axis if axis>=0 else values.ndim+axis
            broadcast=[1]*values.ndim;broadcast[axis]=scales.shape[0]
            scales=scales.reshape(broadcast);zeros=zeros.reshape(broadcast)
        dequantized=(values-zeros)*scales
        replacements.append(numpy_helper.from_array(dequantized.astype(np.float32),node.output[0]))
        for name in node.input:
            if name and consumer_counts.get(name,0)==1:removed_inputs.add(name)
    if not replacements:return 0
    del graph.node[:];graph.node.extend(kept)
    replacement_names={item.name for item in replacements}
    retained=[item for item in graph.initializer if item.name not in removed_inputs and item.name not in replacement_names]
    del graph.initializer[:];graph.initializer.extend(retained);graph.initializer.extend(replacements)
    onnx.checker.check_model(model)
    onnx.save(model,str(destination))
    return len(replacements)


def replace_3d_unsupported_qdq_with_fake_quant(path: str | Path) -> int:
    """Preserve quantization around unsupported 3D ops without Q/DQ fusion.

    TensorRT 10.3 incorrectly propagates an output QuantizeLinear constraint
    through 3D MaxPool/Resize/Concat and then finds no legal INT8 format.
    Expressing adjacent input/output Q/DQ pairs as the same floating-point
    round, clip and scale operations keeps these ops in FP16 while retaining
    the calibrated activation grid.
    """

    try:
        import onnx
        from onnx import helper,numpy_helper
    except ImportError as exc:
        raise ImportError("onnx is required to make 3D Q/DQ graphs TensorRT-compatible") from exc
    destination=Path(path);model=onnx.load(str(destination));graph=model.graph
    initializer_names={item.name for item in graph.initializer}
    graph_inputs=[value for value in graph.input if value.name not in initializer_names]
    if not graph_inputs or len(graph_inputs[0].type.tensor_type.shape.dim)!=5:return 0
    initializers={item.name:item for item in graph.initializer}
    consumers={};producers={}
    for node in graph.node:
        for name in node.input:consumers.setdefault(name,[]).append(node)
        for name in node.output:producers[name]=node
    target_pairs={}
    for protected in graph.node:
        if protected.op_type not in {"MaxPool","Resize","Concat"} or len(protected.output)!=1:continue
        # Downstream Q/DQ can propagate an INT8 output requirement into the
        # protected operator.
        protected_consumers=consumers.get(protected.output[0],[])
        if len(protected_consumers)==1 and protected_consumers[0].op_type=="QuantizeLinear":
            quantize=protected_consumers[0];quantize_consumers=consumers.get(quantize.output[0],[])
            if len(quantize_consumers)==1 and quantize_consumers[0].op_type=="DequantizeLinear":
                target_pairs[quantize.output[0]]=(quantize,quantize_consumers[0])
        # Upstream Q/DQ can propagate INT8 through activations and into the
        # protected operator's input.
        data_inputs=list(protected.input) if protected.op_type=="Concat" else list(protected.input[:1])
        for data_input in data_inputs:
            upstream=producers.get(data_input)
            while upstream is not None and upstream.op_type in {"Relu","LeakyRelu","Clip","Identity"} and upstream.input:
                upstream=producers.get(upstream.input[0])
            if upstream is not None and upstream.op_type=="DequantizeLinear" and upstream.input:
                quantize=producers.get(upstream.input[0])
                if quantize is not None and quantize.op_type=="QuantizeLinear":
                    target_pairs[quantize.output[0]]=(quantize,upstream)
    replacements={};skip=set();new_initializers=[]
    for quantize,dequantize in target_pairs.values():
        if len(quantize.input)<3 or list(quantize.input[1:])!=list(dequantize.input[1:]):continue
        scale=initializers.get(quantize.input[1]);zero=initializers.get(quantize.input[2])
        if scale is None or zero is None:continue
        scale_values=numpy_helper.to_array(scale)
        zero_values=numpy_helper.to_array(zero)
        if scale_values.ndim!=0 or zero_values.ndim!=0:continue
        if zero.data_type==onnx.TensorProto.INT8:qmin,qmax=-128.0,127.0
        elif zero.data_type==onnx.TensorProto.UINT8:qmin,qmax=0.0,255.0
        else:continue
        prefix=quantize.name or quantize.output[0]
        zero_float=f"{prefix}_zero_float";minimum=f"{prefix}_minimum_float";maximum=f"{prefix}_maximum_float"
        divided=f"{prefix}_FloatDiv";shifted=f"{prefix}_FloatShift";rounded=f"{prefix}_FloatRound"
        clipped=f"{prefix}_FloatClip";centered=f"{prefix}_FloatCenter"
        new_initializers.extend((
            numpy_helper.from_array(np.asarray(zero_values,dtype=np.float32),zero_float),
            numpy_helper.from_array(np.asarray(qmin,dtype=np.float32),minimum),
            numpy_helper.from_array(np.asarray(qmax,dtype=np.float32),maximum),
        ))
        replacements[quantize.output[0]]=[
            helper.make_node("Div",[quantize.input[0],quantize.input[1]],[divided],name=f"{prefix}_FloatDivNode"),
            helper.make_node("Add",[divided,zero_float],[shifted],name=f"{prefix}_FloatShiftNode"),
            helper.make_node("Round",[shifted],[rounded],name=f"{prefix}_FloatRoundNode"),
            helper.make_node("Clip",[rounded,minimum,maximum],[clipped],name=f"{prefix}_FloatClipNode"),
            helper.make_node("Sub",[clipped,zero_float],[centered],name=f"{prefix}_FloatCenterNode"),
            helper.make_node("Mul",[centered,quantize.input[1]],[dequantize.output[0]],name=f"{prefix}_FloatScaleNode"),
        ]
        skip.add(dequantize.output[0])
    if not replacements:return 0
    nodes=[]
    for node in graph.node:
        if node.op_type=="QuantizeLinear" and node.output and node.output[0] in replacements:
            nodes.extend(replacements[node.output[0]])
        elif not (node.op_type=="DequantizeLinear" and node.output and node.output[0] in skip):
            nodes.append(node)
    del graph.node[:];graph.node.extend(nodes)
    graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model);onnx.save(model,str(destination))
    return len(replacements)


def onnx_qdq_int8(input_path,output_path,calibration_reader,nodes_to_quantize=None):
    """Create a symmetric calibrated Q/DQ graph suitable for TensorRT INT8."""

    try:
        from onnxruntime.quantization import CalibrationMethod,QuantFormat,QuantType,quantize_static
    except ImportError as exc:raise ImportError("onnxruntime quantization is required for Q/DQ export") from exc
    destination=Path(output_path);destination.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(calibration_reader,"rewind"):calibration_reader.rewind()
    quantize_static(
        str(input_path),str(destination),calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv","Gemm","MatMul"],
        nodes_to_quantize=list(nodes_to_quantize or []),
        extra_options={
            "ActivationSymmetric":True,
            "WeightSymmetric":True,
        },
    )
    folded=fold_int32_dequantize_constants(destination)
    precision_barriers=replace_3d_unsupported_qdq_with_fake_quant(destination)
    logging.getLogger("racmed.pipeline").info(
        "QDQ_TENSORRT_COMPATIBILITY | artifact=%s folded_int32_biases=%d precision_barriers=%d",
        destination.name,folded,precision_barriers,
    )
    return destination


def _onnx_node_matches_layer(node_name: str, layer_name: str) -> bool:
    """Match a PyTorch module path to its corresponding exported ONNX node path."""

    def canonical_parts(value: str, *, exported: bool) -> list[str]:
        parts=[part for part in value.replace(".","/").split("/") if part]
        if exported and parts and parts[0]=="model":
            parts=parts[1:]
        if exported and parts and parts[-1] in {"Conv","Gemm","MatMul"}:
            parts=parts[:-1]
        collapsed=[];index=0
        while index<len(parts):
            if (
                exported
                and index+5<len(parts)
                and parts[index]==parts[index+1]==parts[index+3]
                and parts[index+2]==parts[index+4]
            ):
                # torch.onnx may render ModuleList paths such as
                # blocks/blocks.0/blocks.0.1; recover blocks/0/1.
                collapsed.extend((parts[index],parts[index+2],parts[index+5]))
                index+=6
                continue
            collapsed.append(parts[index]);index+=1
        return collapsed

    node_parts = canonical_parts(node_name,exported=True)
    layer_parts = canonical_parts(layer_name,exported=False)
    if not layer_parts or len(layer_parts) > len(node_parts):
        return False
    width = len(layer_parts)
    return any(node_parts[index : index + width] == layer_parts for index in range(len(node_parts) - width + 1))


def onnx_selective_int8(input_path, output_path, int8_layers: Iterable[str]) -> dict:
    """Execute dynamic INT8 only for ONNX nodes selected by a layer-risk policy."""

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:
        raise ImportError("onnx and onnxruntime are required for selective ONNX INT8 quantization") from exc

    source = Path(input_path)
    destination = Path(output_path)
    requested = [str(layer) for layer in int8_layers]
    graph = onnx.load(source).graph
    quantizable = [node for node in graph.node if node.op_type in {"Conv", "Gemm", "MatMul"}]
    selected = [
        node.name
        for node in quantizable
        if any(_onnx_node_matches_layer(node.name, layer) for layer in requested)
    ]
    unmatched = [
        layer
        for layer in requested
        if not any(_onnx_node_matches_layer(node.name, layer) for node in quantizable)
    ]
    if not selected:
        raise ValueError(
            "Risk-aware INT8 policy did not match any quantizable ONNX nodes; "
            f"requested layers={requested[:5]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = destination.with_name(f".{destination.stem}.prepared.onnx")
    quant_pre_process(source, prepared, skip_symbolic_shape=True)
    prepared_graph = onnx.load(prepared).graph
    runtime_nodes = [
        node.name
        for node in prepared_graph.node
        if node.op_type in {"Conv", "Gemm", "MatMul"}
        and any(_onnx_node_matches_layer(node.name, layer) for layer in requested)
    ]
    quantize_dynamic(
        str(prepared),
        str(destination),
        weight_type=QuantType.QInt8,
        nodes_to_quantize=runtime_nodes,
    )
    prepared.unlink(missing_ok=True)
    quantized_graph = onnx.load(destination).graph
    quantized_types = {"DynamicQuantizeLinear", "MatMulInteger", "ConvInteger", "QLinearConv", "QGemm"}
    quantized_count = sum(node.op_type in quantized_types for node in quantized_graph.node)
    return {
        "mode": "selective_fp32_int8",
        "requested_layers": requested,
        "selected_nodes": selected,
        "runtime_selected_nodes": runtime_nodes,
        "selected_node_count": len(selected),
        "unmatched_layers": unmatched,
        "quantized_node_count": quantized_count,
        "output_path": str(destination),
    }


def onnx_selective_qdq_int8(input_path,output_path,int8_layers:Iterable[str],calibration_reader) -> dict:
    """Create a TensorRT-compatible selective Q/DQ graph from layer risk."""

    try:import onnx
    except ImportError as exc:raise ImportError("onnx is required for selective Q/DQ export") from exc
    requested=[str(layer) for layer in int8_layers]
    quantizable=[node for node in onnx.load(str(input_path)).graph.node if node.op_type in {"Conv","Gemm","MatMul"}]
    selected=[node.name for node in quantizable if any(_onnx_node_matches_layer(node.name,layer) for layer in requested)]
    unmatched=[layer for layer in requested if not any(_onnx_node_matches_layer(node.name,layer) for node in quantizable)]
    if not selected:
        raise ValueError(
            "Risk-aware Q/DQ policy did not match any quantizable ONNX nodes; "
            f"requested layers={requested[:5]}"
        )
    destination=onnx_qdq_int8(input_path,output_path,calibration_reader,selected)
    graph=onnx.load(str(destination)).graph
    return {
        "mode":"selective_qdq_int8",
        "requested_layers":requested,
        "selected_nodes":selected,
        "selected_node_count":len(selected),
        "unmatched_layers":unmatched,
        "quantize_linear_nodes":sum(node.op_type=="QuantizeLinear" for node in graph.node),
        "dequantize_linear_nodes":sum(node.op_type=="DequantizeLinear" for node in graph.node),
        "output_path":str(destination),
    }
