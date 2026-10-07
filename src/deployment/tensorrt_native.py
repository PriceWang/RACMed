"""Native TensorRT engine construction and execution for Jetson deployment."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def _trtexec() -> Path:
    """Locate the JetPack-provided TensorRT command-line builder."""

    candidates=[shutil.which("trtexec"),"/usr/src/tensorrt/bin/trtexec"]
    executable=next((Path(item) for item in candidates if item and Path(item).is_file()),None)
    if executable is None:
        raise FileNotFoundError(
            "TensorRT trtexec was not found. Install the JetPack TensorRT development package."
        )
    return executable


def _sha256(path:Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _shape_spec(values:dict[str,np.ndarray]) -> str:
    return ",".join(f"{name}:{'x'.join(str(int(size)) for size in value.shape)}" for name,value in values.items())


def _dynamic_input_names(path:Path) -> set[str]:
    """Return graph inputs containing at least one symbolic/unknown dimension."""

    try:import onnx
    except ImportError as exc:raise ImportError("onnx is required to inspect TensorRT input shapes") from exc
    graph=onnx.load(str(path),load_external_data=False).graph
    initializer_names={item.name for item in graph.initializer}
    dynamic=set()
    for value in graph.input:
        if value.name in initializer_names:continue
        dimensions=value.type.tensor_type.shape.dim
        if any(dimension.dim_param or not dimension.HasField("dim_value") or dimension.dim_value<=0 for dimension in dimensions):
            dynamic.add(value.name)
    return dynamic


def build_engine(
    onnx_path:str|Path,
    values:dict[str,np.ndarray],
    *,
    int8:bool,
) -> tuple[Path,dict[str,Any]]:
    """Build or reuse a source-hash-validated fixed-shape TensorRT engine."""

    source=Path(onnx_path)
    if source.suffix!=".onnx" or not source.is_file():
        raise FileNotFoundError(f"A source ONNX model is required for TensorRT build: {source}")
    engine=source.with_suffix(".engine")
    metadata_path=engine.with_suffix(engine.suffix+".json")
    source_hash=_sha256(source)
    shapes={name:list(map(int,value.shape)) for name,value in values.items()}
    precision="int8_fp16" if int8 else "fp16"
    expected={"source_sha256":source_hash,"input_shapes":shapes,"precision":precision}
    if engine.is_file() and metadata_path.is_file():
        metadata=json.loads(metadata_path.read_text())
        if all(metadata.get(key)==value for key,value in expected.items()):
            logging.getLogger("racmed.pipeline").info(
                "TENSORRT_ENGINE_CACHE_HIT | artifact=%s precision=%s",engine.name,precision
            )
            return engine,metadata

    executable=_trtexec()
    temporary=engine.with_suffix(engine.suffix+".tmp")
    temporary.unlink(missing_ok=True)
    dynamic_names=_dynamic_input_names(source)
    dynamic_values={name:value for name,value in values.items() if name in dynamic_names}
    shape_spec=_shape_spec(dynamic_values)
    command=[
        str(executable),f"--onnx={source.resolve()}",f"--saveEngine={temporary.resolve()}",
        "--skipInference","--fp16","--profilingVerbosity=detailed",
    ]
    if shape_spec:
        command.extend((f"--minShapes={shape_spec}",f"--optShapes={shape_spec}",f"--maxShapes={shape_spec}"))
    if int8:command.append("--int8")
    timing_cache=source.parent/"tensorrt_native.timing.cache"
    command.append(f"--timingCacheFile={timing_cache.resolve()}")
    logger=logging.getLogger("racmed.pipeline")
    logger.info(
        "TENSORRT_ENGINE_BUILD_START | artifact=%s precision=%s dynamic_shapes=%s",
        source.name,precision,shape_spec or "static",
    )
    subprocess.run(command,check=True)
    if not temporary.is_file() or temporary.stat().st_size==0:
        raise RuntimeError(f"TensorRT reported success but did not create an engine: {temporary}")
    temporary.replace(engine)
    version=subprocess.run([str(executable),"--version"],capture_output=True,text=True,check=False).stdout.splitlines()
    metadata={**expected,"engine":str(engine),"trtexec_version":version[0] if version else "unknown"}
    metadata_path.write_text(json.dumps(metadata,indent=2))
    logger.info("TENSORRT_ENGINE_BUILD_DONE | engine=%s size_bytes=%d",engine.name,engine.stat().st_size)
    return engine,metadata


def _import_tensorrt():
    """Import JetPack's system TensorRT bindings from an isolated conda env."""

    try:
        import tensorrt as trt
        return trt
    except ImportError:
        system_path=Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")
        if system_path.is_dir() and str(system_path) not in sys.path:sys.path.append(str(system_path))
        try:
            import tensorrt as trt
            return trt
        except ImportError as exc:
            raise ImportError(
                "JetPack TensorRT Python bindings are unavailable. Install python3-libnvinfer."
            ) from exc


def _torch_dtype(trt:Any,dtype:Any) -> torch.dtype:
    mapping={
        np.dtype(np.float32):torch.float32,
        np.dtype(np.float16):torch.float16,
        np.dtype(np.int8):torch.int8,
        np.dtype(np.int32):torch.int32,
        np.dtype(np.int64):torch.int64,
        np.dtype(np.bool_):torch.bool,
        np.dtype(np.uint8):torch.uint8,
    }
    converted=np.dtype(trt.nptype(dtype))
    if converted not in mapping:raise TypeError(f"Unsupported TensorRT tensor dtype: {converted}")
    return mapping[converted]


def native_runner(
    engine_path:str|Path,
    values:dict[str,np.ndarray],
    *,
    require_int8:bool,
) -> tuple[Callable[[],None],dict[str,Any],Callable[[],float]]:
    """Load an engine and return a synchronized callable plus layer evidence."""

    if not torch.cuda.is_available():raise RuntimeError("Native TensorRT execution requires CUDA")
    trt=_import_tensorrt()
    logger=trt.Logger(trt.Logger.WARNING)
    free_before,_=torch.cuda.mem_get_info()
    runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
    context=engine.create_execution_context()
    if context is None:raise RuntimeError("TensorRT could not create an execution context")

    tensors:dict[str,torch.Tensor]={}
    input_names=[]
    for index in range(engine.num_io_tensors):
        name=engine.get_tensor_name(index)
        if engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
            if name not in values:raise KeyError(f"TensorRT input {name!r} is missing from benchmark values")
            dtype=_torch_dtype(trt,engine.get_tensor_dtype(name))
            tensor=torch.as_tensor(values[name],device="cuda",dtype=dtype).contiguous()
            engine_shape=tuple(int(size) for size in engine.get_tensor_shape(name))
            if any(size<0 for size in engine_shape):
                if not context.set_input_shape(name,tuple(tensor.shape)):
                    raise RuntimeError(f"TensorRT rejected input shape {name}={tuple(tensor.shape)}")
            elif engine_shape!=tuple(tensor.shape):
                raise RuntimeError(f"TensorRT static input mismatch: {name} expects {engine_shape}, got {tuple(tensor.shape)}")
            tensors[name]=tensor;input_names.append(name)
    for index in range(engine.num_io_tensors):
        name=engine.get_tensor_name(index)
        if name in tensors:continue
        shape=tuple(int(size) for size in context.get_tensor_shape(name))
        if any(size<0 for size in shape):raise RuntimeError(f"Unresolved TensorRT output shape: {name}={shape}")
        tensors[name]=torch.empty(shape,device="cuda",dtype=_torch_dtype(trt,engine.get_tensor_dtype(name)))
    for name,tensor in tensors.items():
        if not context.set_tensor_address(name,int(tensor.data_ptr())):
            raise RuntimeError(f"TensorRT rejected the device address for tensor {name}")

    inspector=engine.create_engine_inspector()
    layer_information=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:layers=json.loads(layer_information)
    except json.JSONDecodeError:layers=[]
    if isinstance(layers,dict):layers=layers.get("Layers",layers.get("layers",[]))
    layer_strings=[json.dumps(layer) for layer in layers] if isinstance(layers,list) else []
    precision_counts=Counter()
    for layer in layer_strings:
        lowered=layer.lower()
        if "int8" in lowered:precision_counts["int8"]+=1
        if "half" in lowered or "fp16" in lowered:precision_counts["fp16"]+=1
        if "float" in lowered or "fp32" in lowered:precision_counts["fp32"]+=1
    if require_int8 and precision_counts["int8"]==0:
        raise RuntimeError("TensorRT engine inspection found no INT8 layer in a Q/DQ benchmark")

    stream=torch.cuda.Stream()
    def run() -> None:
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned failure")
        stream.synchronize()

    run()
    free_after,_=torch.cuda.mem_get_info()
    details={
        "tensorrt_version":str(trt.__version__),
        "tensorrt_engine":str(engine_path),
        "tensorrt_input_names":input_names,
        "tensorrt_layer_count":len(layer_strings),
        "tensorrt_precision_layer_counts":dict(precision_counts),
        "tensorrt_int8_validated":bool(precision_counts["int8"]) if require_int8 else None,
        "tensorrt_engine_device_memory_mb":max(float(free_before-free_after)/1024**2,0.0),
    }
    peak=lambda:details["tensorrt_engine_device_memory_mb"]
    return run,details,peak


def benchmark_runner(
    onnx_path:str|Path,
    values:dict[str,np.ndarray],
    *,
    int8:bool,
) -> tuple[Callable[[],None],str,dict[str,Any],Callable[[],float]]:
    """Build, inspect, and load a native TensorRT benchmark runner."""

    engine,build_metadata=build_engine(onnx_path,values,int8=int8)
    run,runtime_metadata,peak=native_runner(engine,values,require_int8=int8)
    precision="int8" if int8 else "fp16"
    return run,f"tensorrt:{precision}",{**build_metadata,**runtime_metadata},peak
