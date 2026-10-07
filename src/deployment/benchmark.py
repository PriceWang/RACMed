"""Cross-platform latency, memory, throughput, and power probes."""

from __future__ import annotations

import platform
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from collections import Counter
from typing import Any, Callable

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover - Unix deployment targets provide it
    resource = None

from src.pipeline.model_artifacts import example_input_shape, pipeline_model_role
from src.utils.progress import progress_bar


def _current_process_rss_mb() -> float | None:
    """Return current resident memory without adding a runtime dependency."""

    try:
        if platform.system() == "Linux":
            resident_pages=int(Path("/proc/self/statm").read_text().split()[1])
            return float(resident_pages*os.sysconf("SC_PAGE_SIZE")/1024**2)
        value=subprocess.check_output(
            ["ps","-o","rss=","-p",str(os.getpid())],text=True,timeout=2,
        ).strip()
        return float(value)/1024.0
    except (OSError,ValueError,IndexError,subprocess.SubprocessError):
        return None


def _process_peak_rss_mb() -> float | None:
    """Return the process high-water RSS using the platform's native units."""

    try:
        if resource is None:return None
        value=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value/1024**2 if platform.system()=="Darwin" else value/1024.0
    except (AttributeError,ValueError,OSError):
        return None


def system_metadata() -> dict[str, Any]:
    """Return non-identifying hardware/runtime metadata for result records."""

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _inputs(
    config: dict[str, Any],
    model: torch.nn.Module | None,
    role: str,
) -> dict[str, np.ndarray]:
    """Build artifact inputs without requiring the source PyTorch model.

    ONNX and Core ML artifacts carry the executable graph already. Recreating
    the training model on an edge device is unnecessary and can pull in
    training-only dependencies such as timm or MONAI. Prompt-map metadata is
    therefore read from the selected model role when no model is provided.
    """

    shape = example_input_shape(config, role)
    values = {"input": np.random.randn(*shape).astype("float32")}
    role_config = {**config.get("model", {}), **config.get(role, {})}
    accepts_prompts = (
        bool(getattr(model, "accepts_prompts", False))
        if model is not None
        else bool(role_config.get("prompt_aware", False))
    )
    if accepts_prompts:
        default_channels = max(int(role_config.get("num_classes", 2)) - 1, 1)
        channels = int(
            getattr(
                model,
                "prompt_channels",
                role_config.get("prompt_channels", default_channels),
            )
            if model is not None
            else role_config.get("prompt_channels", default_channels)
        )
        values["prompt_maps"] = np.zeros((shape[0], channels, *shape[2:]), dtype="float32")
    return values


def _time_callable(run: Callable[[], Any], warmup: int, repetitions: int) -> dict[str, float]:
    for _ in progress_bar(range(warmup), desc="Benchmark warm-up", total=warmup, unit="run", leave=False):
        run()
    times = []
    for _ in progress_bar(
        range(repetitions),
        desc="Benchmark measurement",
        total=repetitions,
        unit="run",
        leave=True,
    ):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times)),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "throughput_fps": float(1000.0 / np.mean(times)),
    }


def _parse_tegrastats_power(output: str) -> tuple[str | None, list[float]]:
    """Extract board-input power samples from Jetson tegrastats output."""

    for rail in ("VDD_IN", "POM_5V_IN"):
        samples=[
            float(value)/1000.0
            for value in re.findall(rf"\b{rail}\s+([0-9]+(?:\.[0-9]+)?)mW",output)
        ]
        if samples:return rail,samples
    return None,[]


class _TegrastatsSampler:
    """Capture continuous tegrastats output without relying on ``--count``.

    The JetPack 6 tegrastats shipped on some Orin Nano images does not expose
    the ``--count`` option. A temporary output stream plus explicit process
    termination works across both command variants and samples while the
    inference workload is actually running.
    """

    def __init__(self,interval_ms:int=100):
        self.interval_ms=interval_ms
        self._stream=None
        self._process=None

    def start(self) -> None:
        self._stream=tempfile.TemporaryFile(mode="w+t")
        self._process=subprocess.Popen(
            ["tegrastats","--interval",str(self.interval_ms)],
            stdout=self._stream,stderr=subprocess.STDOUT,text=True,
        )

    def stop(self) -> dict[str,Any]:
        if self._process is None or self._stream is None:return {}
        self._process.terminate()
        try:self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill();self._process.wait(timeout=2)
        self._stream.flush();self._stream.seek(0)
        output=self._stream.read();self._stream.close()
        self._process=None;self._stream=None
        rail,samples=_parse_tegrastats_power(output)
        if not samples:return {}
        return {
            "power_watts_mean":float(np.mean(samples)),
            "power_watts_std":float(np.std(samples)),
            "power_watts_min":float(np.min(samples)),
            "power_watts_max":float(np.max(samples)),
            "power_sample_count":len(samples),
            "power_source":f"tegrastats:{rail}",
        }


def _measure_workload_power(
    run:Callable[[],Any],target:str,duration_seconds:float=2.0,
) -> dict[str,Any]:
    """Sample Jetson board power concurrently with repeated inference."""

    if not target.startswith("jetson") or not shutil.which("tegrastats"):return {}
    sampler=_TegrastatsSampler()
    logger=logging.getLogger("racmed.pipeline")
    logger.info("ENERGY_SAMPLING_START | source=tegrastats duration_seconds=%.1f",duration_seconds)
    sampler.start();started=time.perf_counter();inferences=0
    try:
        while time.perf_counter()-started<duration_seconds:
            run();inferences+=1
    finally:
        elapsed=time.perf_counter()-started
        measurement=sampler.stop()
    measurement.update({
        "power_measurement_duration_seconds":float(elapsed),
        "power_measurement_inferences":inferences,
    })
    logger.info(
        "ENERGY_SAMPLING_DONE | samples=%d mean_watts=%s inferences=%d",
        measurement.get("power_sample_count",0),measurement.get("power_watts_mean"),inferences,
    )
    return measurement


def _onnx_providers(target: str, requested: str = "auto") -> list[str]:
    """Select an explicit ONNX Runtime provider policy.

    TensorRT EP is not an automatic Jetson default: an incompatible ORT,
    TensorRT, plugin, or graph combination can terminate the process instead
    of raising a catchable Python exception. CUDA EP is the safe accelerated
    default; TensorRT remains available as an explicit, separately validated
    choice.
    """

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    aliases={
        "cpu":"CPUExecutionProvider",
        "cuda":"CUDAExecutionProvider",
        "tensorrt":"TensorrtExecutionProvider",
    }
    if requested not in {"auto",*aliases}:
        raise ValueError(f"Unsupported ONNX Runtime provider: {requested}")
    if requested!="auto":
        primary=aliases[requested]
        if primary not in available:
            raise RuntimeError(
                f"Requested ONNX Runtime provider {primary} is unavailable. "
                f"Available providers: {sorted(available)}"
            )
        preferences=[primary]
        if primary=="TensorrtExecutionProvider":preferences.append("CUDAExecutionProvider")
        if primary!="CPUExecutionProvider":preferences.append("CPUExecutionProvider")
    elif target=="jetson-cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "jetson-cuda requires CUDAExecutionProvider for the safe default. "
                "Install the JetPack-compatible ONNX Runtime GPU build, or explicitly "
                "request --onnx-provider tensorrt after validating that provider. "
                f"Available providers: {sorted(available)}"
            )
        preferences=["CUDAExecutionProvider","CPUExecutionProvider"]
    elif target in {"jetson-cpu","raspi5"}:
        preferences=["CPUExecutionProvider"]
    else:
        preferences=["CUDAExecutionProvider","CoreMLExecutionProvider","CPUExecutionProvider"]
    selected=list(dict.fromkeys(provider for provider in preferences if provider in available))
    if not selected:
        raise RuntimeError(f"No usable ONNX Runtime provider. Available providers: {sorted(available)}")
    return selected


def _onnx_quantization_format(path: Path) -> str:
    """Classify ONNX quantization encoding before selecting a GPU backend."""

    try:
        import onnx
    except ImportError as exc:
        raise ImportError("onnx is required to validate GPU deployment artifacts") from exc
    operator_quantized={"ConvInteger","MatMulInteger","QLinearConv","QLinearMatMul","QGemm"}
    operations={node.op_type for node in onnx.load(str(path),load_external_data=False).graph.node}
    if operations & operator_quantized:return "quantized_operator"
    if {"QuantizeLinear","DequantizeLinear"}.issubset(operations):return "qdq"
    return "floating_point"


def _validate_tensorrt_qdq(path: Path) -> dict[str, int]:
    """Reject Q/DQ tensor types unsupported by the JetPack 6 TensorRT parser."""

    import onnx

    graph=onnx.load(str(path),load_external_data=False).graph
    tensor_types={item.name:int(item.data_type) for item in graph.initializer}
    for node in graph.node:
        if node.op_type!="Constant" or not node.output:
            continue
        value=next((attribute.t for attribute in node.attribute if attribute.name=="value" and attribute.HasField("t")),None)
        if value is not None:tensor_types[node.output[0]]=int(value.data_type)
    qdq=[node for node in graph.node if node.op_type in {"QuantizeLinear","DequantizeLinear"}]
    int32_dq=[
        node.name for node in qdq
        if node.op_type=="DequantizeLinear" and tensor_types.get(node.input[0])==onnx.TensorProto.INT32
    ]
    uint8_zero_points=[
        node.name for node in qdq
        if len(node.input)>2 and tensor_types.get(node.input[2])==onnx.TensorProto.UINT8
    ]
    if int32_dq:
        raise RuntimeError(
            "TensorRT Q/DQ artifact contains INT32 DequantizeLinear bias tensors "
            f"({len(int32_dq)} nodes). Re-export it with QuantizeBias=False before "
            "running a Jetson INT8 benchmark."
        )
    if uint8_zero_points:
        raise RuntimeError(
            "TensorRT Q/DQ artifact contains UINT8 zero points "
            f"({len(uint8_zero_points)} nodes). Re-export symmetric QInt8 Q/DQ tensors."
        )
    return {
        "qdq_node_count":len(qdq),
        "int32_dequantize_nodes":0,
        "uint8_zero_point_nodes":0,
    }


def _summarize_ort_profile(profile_path: str | Path) -> dict[str, Any]:
    """Summarize actual execution-provider assignment from one profiled run."""

    path=Path(profile_path)
    try:events=json.loads(path.read_text())
    finally:path.unlink(missing_ok=True)
    nodes=[event for event in events if event.get("cat")=="Node" and event.get("args",{}).get("provider")]
    providers=Counter(event["args"]["provider"] for event in nodes)
    provider_ops:dict[str,Counter]={}
    durations=Counter()
    fallback_compute=[]
    fallback_quantization=[]
    compute_ops={"Conv","FusedConv","NhwcConv","Gemm","MatMul","Attention","QLinearConv","QGemm"}
    for event in nodes:
        assigned=event["args"]["provider"]
        operation=event["args"].get("op_name","unknown")
        provider_ops.setdefault(assigned,Counter())[operation]+=1
        durations[assigned]+=float(event.get("dur",0.0))/1000.0
        if assigned!="TensorrtExecutionProvider" and operation in compute_ops:
            fallback_compute.append(event.get("name",operation))
        if assigned!="TensorrtExecutionProvider" and operation in {"QuantizeLinear","DequantizeLinear"}:
            fallback_quantization.append(event.get("name",operation))
    return {
        "profile_provider_event_counts":dict(providers),
        "profile_provider_op_counts":{provider:dict(counts) for provider,counts in provider_ops.items()},
        "profile_provider_duration_ms":dict(durations),
        "tensorrt_profile_validated":providers.get("TensorrtExecutionProvider",0)>0,
        "tensorrt_fallback_compute_event_count":len(fallback_compute),
        "tensorrt_fallback_compute_events":fallback_compute[:20],
        "tensorrt_fallback_quantization_event_count":len(fallback_quantization),
        "tensorrt_fallback_quantization_events":fallback_quantization[:20],
    }


def _benchmark_onnx(path: Path, values: dict[str, np.ndarray], target: str, provider: str) -> tuple[Callable[[], Any], str, list[str], dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for ONNX deployment") from exc
    providers = _onnx_providers(target,provider)
    encoding=_onnx_quantization_format(path) if target=="jetson-cuda" else "unchecked"
    if encoding=="quantized_operator":
        raise RuntimeError(
            "This ONNX artifact uses ConvInteger/MatMulInteger-style dynamic quantization, "
            "which is a CPU deployment format and is not a valid TensorRT/CUDA INT8 benchmark. "
            "Run it with --target jetson-cpu --onnx-provider cpu. For Jetson GPU INT8, "
            "export a calibrated Q/DQ artifact and use TensorRT."
        )
    if encoding=="qdq" and providers[0]!="TensorrtExecutionProvider":
        raise RuntimeError(
            "A Q/DQ INT8 artifact on Jetson GPU must use --onnx-provider tensorrt; "
            "CUDA EP does not establish a TensorRT INT8 execution claim."
        )
    compatibility={}
    if encoding=="qdq" and providers[0]=="TensorrtExecutionProvider":
        compatibility=_validate_tensorrt_qdq(path)
    provider_specs:list[Any]=providers
    if providers[0]=="TensorrtExecutionProvider":
        cache=path.parent/"trt_engine_cache"
        cache.mkdir(parents=True,exist_ok=True)
        tensorrt_options={
            "trt_int8_enable":encoding=="qdq",
            "trt_fp16_enable":True,
            "trt_engine_cache_enable":True,
            "trt_engine_cache_path":str(cache.resolve()),
            "trt_timing_cache_enable":True,
        }
        workspace=os.environ.get("RACMED_TRT_MAX_WORKSPACE_SIZE")
        if workspace:
            tensorrt_options["trt_max_workspace_size"]=int(workspace)
        provider_specs=[
            ("TensorrtExecutionProvider",tensorrt_options),
            *providers[1:],
        ]
    logger=logging.getLogger("racmed.pipeline")
    logger.info(
        "ONNX_SESSION_BUILD_START | artifact=%s requested_provider=%s provider_order=%s encoding=%s",
        path.name,provider,",".join(providers),encoding,
    )
    session_options=None
    if providers[0]=="TensorrtExecutionProvider":
        session_options=ort.SessionOptions()
        session_options.enable_profiling=True
        session_options.profile_file_prefix=str(path.parent/f".{path.stem}.{os.getpid()}.ort_profile")
    session = ort.InferenceSession(str(path), sess_options=session_options,providers=provider_specs)
    names = [item.name for item in session.get_inputs()]
    feed = {name: values[name] for name in names}
    active=session.get_providers()
    logger.info("ONNX_SESSION_BUILD_DONE | artifact=%s active_providers=%s",path.name,",".join(active))
    profile={}
    if providers[0]=="TensorrtExecutionProvider":
        session.run(None,feed)
        profile=_summarize_ort_profile(session.end_profiling())
        if not profile["tensorrt_profile_validated"]:
            raise RuntimeError("TensorRT was requested but the profiled inference executed no TensorRT node.")
        if encoding=="qdq" and (
            profile["tensorrt_fallback_compute_event_count"]
            or profile["tensorrt_fallback_quantization_event_count"]
        ):
            raise RuntimeError(
                "TensorRT INT8 profiled inference fell back for compute/QDQ nodes: "
                + ", ".join(
                    (profile["tensorrt_fallback_compute_events"]+profile["tensorrt_fallback_quantization_events"])[:5]
                )
            )
        logger.info(
            "ONNX_TENSORRT_PROFILE | provider_events=%s fallback_compute=%d",
            profile["profile_provider_event_counts"],profile["tensorrt_fallback_compute_event_count"],
        )
    return lambda: session.run(None, feed), f"onnxruntime:{active[0]}", active,{**compatibility,**profile}


def _benchmark_coreml(path: Path, values: dict[str, np.ndarray]) -> tuple[Callable[[], Any], str]:
    if platform.system() != "Darwin":
        raise RuntimeError("CoreML execution is available only on macOS")
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError("Install coremltools on Apple Silicon to benchmark CoreML") from exc
    model = ct.models.MLModel(str(path))
    return lambda: model.predict(values), "coreml:mlprogram"


def _benchmark_pytorch(model: torch.nn.Module, values: dict[str, np.ndarray], target: str) -> tuple[Callable[[], Any], str, Callable[[], float]]:
    use_cuda = target == "jetson-cuda" or (target == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tensors = {name: torch.from_numpy(value).to(device) for name, value in values.items()}

    def run() -> None:
        with torch.inference_mode():
            if "prompt_maps" in tensors:
                model.forward_with_prompt_maps(tensors["input"], tensors["prompt_maps"], return_features=False)
            else:
                model(tensors["input"], return_features=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak = lambda: float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    return run, f"pytorch:{device.type}", peak


def benchmark_artifact(
    model: torch.nn.Module | None,
    config: dict[str, Any],
    artifact: str | Path | None = None,
    backend: str = "auto",
    target: str = "auto",
    model_role: str | None = None,
    onnx_provider: str = "auto",
    measure_power: bool = False,
    measure_latency: bool = True,
    measure_memory: bool = True,
) -> dict[str, Any]:
    """Benchmark PyTorch, ONNX Runtime, TensorRT, or CoreML with one protocol."""

    process_baseline=_current_process_rss_mb()
    role = model_role or pipeline_model_role(config, "deployment")
    values = _inputs(config, model, role)
    path = Path(artifact) if artifact else None
    selected = backend
    if selected == "auto":
        if target == "apple-m4" and path and path.suffix in {".mlpackage", ".mlmodel"}:
            selected = "coreml"
        elif target == "jetson-cuda" and path and path.suffix == ".onnx":
            try:
                import onnxruntime as ort

                accelerated = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
                selected = "onnx" if accelerated.intersection(ort.get_available_providers()) else "pytorch"
            except ImportError:
                selected = "pytorch"
        elif path and path.suffix == ".onnx":
            selected = "onnx"
        else:
            selected = "pytorch"
    peak_memory: Callable[[], float] = lambda: 0.0
    backend_details:dict[str,Any]={}
    if selected == "onnx":
        if path is None:
            raise ValueError("An ONNX artifact is required for backend=onnx")
        if not path.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {path}")
        run, backend_name,active_providers,provider_profile = _benchmark_onnx(path, values, target,onnx_provider)
        backend_details={"onnx_provider_requested":onnx_provider,"onnx_providers_active":active_providers,**provider_profile}
    elif selected == "tensorrt":
        if path is None or not path.is_file():
            raise FileNotFoundError(f"A source ONNX artifact is required for TensorRT: {path}")
        if target!="jetson-cuda":raise RuntimeError("Native TensorRT backend requires target=jetson-cuda")
        from src.deployment.tensorrt_native import benchmark_runner
        encoding=_onnx_quantization_format(path)
        run,backend_name,backend_details,peak_memory=benchmark_runner(path,values,int8=encoding=="qdq")
    elif selected == "coreml":
        if path is None:
            raise ValueError("A CoreML artifact is required for backend=coreml")
        if not path.exists():
            raise FileNotFoundError(
                f"Core ML artifact not found: {path}. Run pipeline step 5 with --coreml first."
            )
        run, backend_name = _benchmark_coreml(path, values)
    elif selected == "pytorch":
        if model is None:
            raise ValueError("A PyTorch model is required for backend=pytorch")
        run, backend_name, peak_memory = _benchmark_pytorch(model, values, target)
    else:
        raise ValueError(f"Unsupported benchmark backend: {selected}")
    deploy = config.get("deployment", {})
    result=(
        _time_callable(run,int(deploy.get("warmup",5)),int(deploy.get("repetitions",30)))
        if measure_latency else {}
    )
    if measure_memory and not measure_latency and not measure_power:
        # A memory-only probe still executes the graph once so lazy runtime
        # allocations are represented without repeating the timing protocol.
        run()
    if measure_power:
        duration=float(deploy.get("energy_duration_seconds",2.0))
        result.update(_measure_workload_power(run,target,duration))
    accelerator_peak=float(peak_memory())
    process_peak=_process_peak_rss_mb()
    if accelerator_peak>0:
        reported_peak=accelerator_peak;memory_kind="accelerator_allocated"
    else:
        reported_peak=float(process_peak or 0.0);memory_kind="process_max_rss"
    incremental=(
        max(float(process_peak)-float(process_baseline),0.0)
        if process_peak is not None and process_baseline is not None else None
    )
    result.update({
        "backend":backend_name,"target":target,
        "peak_memory_mb":reported_peak,"peak_memory_kind":memory_kind,
        "process_baseline_rss_mb":process_baseline,
        "process_peak_rss_mb":process_peak,
        "process_incremental_peak_rss_mb":incremental,
        "system":system_metadata(),**backend_details,
    })
    return result


def _power_probe(target: str, backend: str) -> tuple[float | None, str | None]:
    gpu_execution=("cuda" in backend.lower() or "tensorrt" in backend.lower() or target=="jetson-cuda")
    if gpu_execution and shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            return float(output.splitlines()[0]), "nvidia-smi"
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            logging.getLogger(__name__).debug("nvidia-smi power probe unavailable: %s",exc)
    if target.startswith("jetson") and shutil.which("tegrastats"):
        sampler=_TegrastatsSampler();sampler.start()
        try:
            time.sleep(.2);measurement=sampler.stop()
            if measurement.get("power_watts_mean") is not None:
                return measurement["power_watts_mean"],measurement["power_source"]
        except (OSError,subprocess.SubprocessError) as exc:
            logging.getLogger(__name__).debug("tegrastats power probe unavailable: %s",exc)
    return None, None


def benchmark_energy(target: str, latency: dict[str, Any]) -> dict[str, Any]:
    """Attach a best-effort platform power sample without aborting latency runs."""

    backend=str(latency.get("backend","unknown"))
    watts=latency.get("power_watts_mean")
    source=latency.get("power_source")
    if watts is None:watts,source=_power_probe(target,backend)
    warning = None
    if watts is None:
        warning = f"Energy telemetry matching backend '{backend}' is unavailable; latency results remain valid."
        warnings.warn(warning)
    return {
        "energy_available": watts is not None,
        "power_watts": watts,
        "power_watts_std":latency.get("power_watts_std"),
        "power_watts_min":latency.get("power_watts_min"),
        "power_watts_max":latency.get("power_watts_max"),
        "power_sample_count":latency.get("power_sample_count"),
        "power_measurement_duration_seconds":latency.get("power_measurement_duration_seconds"),
        "power_measurement_inferences":latency.get("power_measurement_inferences"),
        "source": source,
        "estimated_energy_joules_per_inference": watts * latency["latency_mean_ms"] / 1000.0 if watts is not None else None,
        "warning": warning,
        "latency": latency,
    }
