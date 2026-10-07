"""Hardware benchmark stage for PyTorch, ONNX Runtime, and CoreML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.deployment.benchmark import benchmark_artifact, benchmark_energy
from src.engine.checkpoint import load_model_checkpoint
from src.models.model_factory import create_model
from src.pipeline.model_artifacts import export_config, pipeline_checkpoint, pipeline_model_role
from src.pipeline.progress import pipeline_logger


def run_benchmark(
    config: dict[str, Any],
    artifact: str | Path | None = None,
    backend: str = "auto",
    target: str = "auto",
    energy: bool = False,
    output_label: str | None = None,
    onnx_provider: str = "auto",
    energy_only: bool = False,
    memory_only: bool = False,
) -> dict[str, Any]:
    """Benchmark the selected deployment artifact and persist a JSON record."""

    cfg = config
    out = Path(cfg.get("output_dir", "outputs/experiment"))
    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    role = pipeline_model_role(cfg, "deployment")
    model_cfg = export_config(cfg, role)
    # Executable artifacts do not need their source PyTorch model. Keeping this
    # path artifact-only avoids importing training/backbone dependencies on
    # constrained deployment devices. ``auto`` may still select PyTorch (for
    # example, Jetson CUDA fallback), so it retains the model-loading path.
    model = None
    if backend not in {"onnx", "coreml", "tensorrt"}:
        model = create_model(model_cfg, role)
        checkpoint = pipeline_checkpoint(cfg, out, "deployment")
        if checkpoint is not None:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Deployment checkpoint not found: {checkpoint}")
            model,_=load_model_checkpoint(checkpoint,model)
    target_name = target.replace("-", "_")
    if output_label:
        filename=f"latency_{output_label}_{target_name}.json" if target!="auto" else f"latency_{output_label}.json"
    else:
        filename = f"latency_{target_name}.json" if target != "auto" else "latency.json"
    latency_path=deployment/filename
    if energy_only and memory_only:raise ValueError("energy_only and memory_only are mutually exclusive")
    if energy_only and not energy:raise ValueError("energy_only requires energy=True")
    if energy_only or memory_only:
        if not latency_path.is_file():
            raise FileNotFoundError(f"Supplemental sampling requires an existing latency record: {latency_path}")
        formal_latency=json.loads(latency_path.read_text())
    result = benchmark_artifact(
        model,model_cfg,artifact,backend,target,role,onnx_provider,
        measure_power=energy and not memory_only,
        measure_latency=not (energy_only or memory_only),
        measure_memory=True,
    )
    if energy_only:
        supplemental={key:value for key,value in result.items() if key.startswith("power_") or "memory" in key or "rss" in key}
        result={**formal_latency,**supplemental}
    elif memory_only:
        supplemental={key:value for key,value in result.items() if "memory" in key or "rss" in key}
        result={**formal_latency,**supplemental}
        latency_path.write_text(json.dumps(result,indent=2))
        energy_name=(
            f"energy_{output_label}_{target_name}.json" if output_label and target!="auto"
            else f"energy_{output_label}.json" if output_label
            else f"energy_{target_name}.json" if target!="auto" else "energy.json"
        )
        energy_path=deployment/energy_name
        if energy_path.exists():
            payload=json.loads(energy_path.read_text());payload["latency"]=result
            energy_path.write_text(json.dumps(payload,indent=2))
    else:
        latency_path.write_text(json.dumps(result, indent=2))
    pipeline_logger().info(
        "BENCHMARK_RESULT | backend=%s target=%s mean_ms=%.3f p95_ms=%.3f throughput_fps=%.2f",
        result["backend"],
        target,
        result["latency_mean_ms"],
        result["latency_p95_ms"],
        result["throughput_fps"],
    )
    if energy and not memory_only:
        energy_result = benchmark_energy(target, result)
        if output_label:
            energy_name=f"energy_{output_label}_{target_name}.json" if target!="auto" else f"energy_{output_label}.json"
        else:
            energy_name = f"energy_{target_name}.json" if target != "auto" else "energy.json"
        (deployment / energy_name).write_text(json.dumps(energy_result, indent=2))
        pipeline_logger().info(
            "ENERGY_RESULT | available=%s source=%s power_watts=%s",
            energy_result["energy_available"],
            energy_result["source"],
            energy_result["power_watts"],
        )
        return energy_result
    return result


def run_onnx_benchmark_suite(
    config:dict[str,Any],artifacts:dict[str,str|Path],target:str="auto"
) -> pd.DataFrame:
    """Benchmark all ONNX precision variants under one timing protocol."""

    rows=[]
    for label in ("onnx_fp32","onnx_uniform_int8","onnx_risk_aware_int8"):
        path=artifacts.get(label)
        if path is None or not Path(path).exists():
            continue
        pipeline_logger().info("BENCHMARK_VARIANT_START | backend=%s artifact=%s",label,Path(path).name)
        result=run_benchmark(config,path,"onnx",target,energy=False,output_label=label)
        rows.append({"backend_variant":label,"artifact":str(path),**result})
        summary=Path(config.get("output_dir","outputs/experiment"))/"deployment"/"results"/label/"summary.csv"
        if summary.exists():
            frame=pd.read_csv(summary)
            for key in ("latency_mean_ms","latency_std_ms","latency_p50_ms","latency_p95_ms","throughput_fps","peak_memory_mb"):
                frame[key]=result[key]
            frame.to_csv(summary,index=False)
    output=Path(config.get("output_dir","outputs/experiment"))/"deployment"/"latency_comparison.csv"
    table=pd.DataFrame(rows)
    table.to_csv(output,index=False)
    if rows:
        preferred=next((row for row in rows if row["backend_variant"]=="onnx_risk_aware_int8"),rows[0])
        primary_result={key:value for key,value in preferred.items() if key not in {"backend_variant","artifact"}}
        (output.parent/"latency.json").write_text(json.dumps(primary_result,indent=2))
    return table
