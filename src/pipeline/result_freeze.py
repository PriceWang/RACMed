"""Freeze final experiment evidence into a portable integrity manifest.

The manifest hashes source/configuration and text-based result artifacts while
inventorying large model binaries by path and size. This keeps the normal
analysis stage fast and avoids embedding machine-specific paths in evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import yaml


_CORE_PACKAGES = (
    "torch",
    "torchvision",
    "monai",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "onnx",
    "onnxruntime",
    "timm",
)

_REAL_DEVICE_PROTOCOL = {
    "apple-m4": (
        "onnx_fp32_cpu",
        "onnx_uniform_int8_cpu",
        "onnx_risk_aware_int8_cpu",
        "coreml",
    ),
    "raspi5": (
        "onnx_fp32_cpu",
        "onnx_uniform_int8_cpu",
        "onnx_risk_aware_int8_cpu",
    ),
    "jetson-cpu": (
        "onnx_fp32_cpu",
        "onnx_uniform_int8_cpu",
        "onnx_risk_aware_int8_cpu",
    ),
    "jetson-cuda": (
        "onnx_fp32_cuda",
        "onnx_fp16_tensorrt",
        "tensorrt_native_uniform_qdq_int8",
        "tensorrt_native_risk_aware_qdq_int8",
    ),
}

_MANIFEST_EXCLUSIONS = {
    "result_artifact_manifest.csv",
    "result_manifest.json",
}

_PRIVATE_PATH = re.compile(r"(?:^|[\s\"'=,:])/(?:home|Users|scratch)/[^\s\"',]*")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, anchor: Path) -> str:
    """Return a POSIX relative path without exposing an unrelated host path."""

    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return path.name


def _aggregate_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _file_records(paths: Iterable[Path], project_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        records.append(
            {
                "path": _relative(path, project_root),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _source_files(project_root: Path) -> list[Path]:
    files = list((project_root / "src").rglob("*.py"))
    files.extend((project_root / "scripts").glob("*.py"))
    files.extend((project_root / "configs" / "tasks").rglob("*.yaml"))
    for name in (
        "README.md",
        "environment.yml",
        "requirements.txt",
        "requirements-optional.txt",
        "requirements-jetson.txt",
        "full_pipeline.sh",
        "real_device.sh",
    ):
        path = project_root / name
        if path.is_file():
            files.append(path)
    return files


def _result_files(results_root: Path) -> list[Path]:
    exact_names = {
        "summary.csv",
        "metrics.json",
        "per_case_metrics.csv",
        "config.yaml",
        "package_versions.json",
        "reproduction_gate.json",
        "compression_metrics.json",
        "layer_risk_scores.csv",
        "elastic_pruning_policy.json",
        "pruning_policy.json",
        "mixed_precision_policy.json",
        "artifacts.json",
    }
    selected: list[Path] = []
    analysis = results_root / "analysis"
    for path in results_root.rglob("*"):
        if not path.is_file() or path.name in _MANIFEST_EXCLUSIONS:
            continue
        is_analysis = path.parent == analysis and path.suffix.lower() in {
            ".csv",
            ".json",
            ".md",
            ".png",
            ".xml",
        }
        is_measurement = (
            path.name in exact_names
            or path.name.startswith("lesion_operating_curve")
            or path.name.startswith("latency_")
            or path.name.startswith("energy_")
        )
        if is_analysis or is_measurement:
            selected.append(path)
    return selected


def _binary_inventory(results_root: Path, project_root: Path) -> dict[str, Any]:
    suffixes = {".pt", ".pth", ".onnx", ".engine", ".plan", ".mlmodel"}
    files = [path for path in results_root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    return {
        "hash_policy": "inventory_only; use an archival checksum tool to freeze model binaries",
        "count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "paths": [_relative(path, project_root) for path in sorted(files)],
    }


def _load_task_configs(project_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    configs = []
    for path in sorted((project_root / "configs" / "tasks").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        if payload.get("pipeline") is not None:
            configs.append((path, payload))
    return configs


def _experiment_root(results_root: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("output_dir", config.get("experiment_name", "experiment"))))
    return results_root / configured.name


def _expected_paths(
    results_root: Path,
    configs: list[tuple[Path, dict[str, Any]]],
) -> dict[str, list[Path]]:
    expected: dict[str, list[Path]] = {
        "teacher_reproduction_gates": [],
        "method_run_summaries": [],
        "training_variant_summaries": [],
        "compression_summaries": [],
        "quantization_summaries": [],
        "lesion_froc_summaries": [],
    }
    teacher_gates: set[Path] = set()
    for _, config in configs:
        pipeline = config.get("pipeline", {})
        experiment = _experiment_root(results_root, config)
        seeds = pipeline.get("seeds", [config.get("seed", 0)])
        teacher_dir = config.get("teacher_output_dir")
        if pipeline.get("evaluate_teacher", True) and teacher_dir:
            configured_root=Path(str(config.get("output_dir","")))
            configured_teacher=Path(str(teacher_dir))
            try:relative_teacher=configured_teacher.relative_to(configured_root)
            except ValueError:relative_teacher=Path("teacher")/"native"
            teacher_gates.add(experiment/relative_teacher/"reproduction_gate.json")
        for method in pipeline.get("methods", []):
            for seed in seeds:
                expected["method_run_summaries"].append(
                    experiment / "training" / str(method) / f"seed_{seed}" / "summary.csv"
                )
        for variant in pipeline.get("training_variants", []):
            for seed in variant.get("seeds", seeds):
                expected["training_variant_summaries"].append(
                    experiment / "ablations" / str(variant["name"]) / f"seed_{seed}" / "summary.csv"
                )
        for variant in pipeline.get("compression_variants", []):
            variant_root = experiment / "compression" / str(variant["name"])
            variant_seeds = variant.get("seeds", seeds)
            for seed in variant_seeds:
                expected["compression_summaries"].append(variant_root / f"seed_{seed}" / "summary.csv")
            if variant.get("export") and variant.get("evaluate_onnx"):
                artifacts = variant.get("overrides", {}).get("deployment", {}).get(
                    "reliability_artifacts",
                    ("onnx_fp32", "onnx_uniform_qdq_int8", "onnx_risk_aware_qdq_int8"),
                )
                for seed in variant.get("export_seeds", variant_seeds):
                    for artifact in artifacts:
                        expected["quantization_summaries"].append(
                            variant_root
                            / f"seed_{seed}"
                            / "deployment"
                            / "results"
                            / str(artifact)
                            / "summary.csv"
                        )
        if config.get("evaluation", {}).get("lesion_threshold_candidates"):
            methods = pipeline.get(
                "lesion_operating_curve_methods",
                [name for name in ("standalone", "vanilla_kd", "racmed") if name in pipeline.get("methods", [])],
            )
            for method in methods:
                for seed in seeds:
                    expected["lesion_froc_summaries"].append(
                        experiment / "training" / str(method) / f"seed_{seed}" / "lesion_operating_curve_summary.csv"
                    )
    expected["teacher_reproduction_gates"] = sorted(teacher_gates)
    return expected


def _check_row(
    name: str,
    expected: list[Path],
    results_root: Path,
    invalid: Iterable[Path] = (),
) -> dict[str, Any]:
    missing = [path for path in expected if not path.is_file()]
    invalid_paths = sorted(set(invalid))
    notes = []
    if missing:
        notes.append("missing=" + "|".join(_relative(path, results_root) for path in missing))
    if invalid_paths:
        notes.append("invalid=" + "|".join(_relative(path, results_root) for path in invalid_paths))
    return {
        "check": name,
        "expected": len(expected),
        "observed": len(expected) - len(missing),
        "missing": len(missing),
        "invalid": len(invalid_paths),
        "complete": not missing and not invalid_paths,
        "notes": ";".join(notes),
    }


def _compression_flop_check(paths: list[Path], results_root: Path) -> dict[str, Any]:
    metrics = [path.parent / "compression" / "compression_metrics.json" for path in paths]
    invalid = []
    for path in metrics:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid.append(path)
            continue
        if payload.get("baseline_flops") is None or payload.get("flops") is None:
            invalid.append(path)
    return _check_row("compression_flops", metrics, results_root, invalid)


def _teacher_gate_check(paths: list[Path], results_root: Path) -> dict[str, Any]:
    invalid = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            if not bool(json.loads(path.read_text()).get("passed")):
                invalid.append(path)
        except (OSError, json.JSONDecodeError):
            invalid.append(path)
    return _check_row("teacher_reproduction_gates", paths, results_root, invalid)


def _deployment_roots(
    results_root: Path,
    configs: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, tuple[str, ...]]]:
    roots = []
    for _, config in configs:
        pipeline = config.get("pipeline", {})
        targets = tuple(pipeline.get("real_device_targets", _REAL_DEVICE_PROTOCOL))
        unknown = set(targets) - set(_REAL_DEVICE_PROTOCOL)
        if unknown:
            raise ValueError(f"Unknown real-device targets: {sorted(unknown)}")
        experiment = _experiment_root(results_root, config)
        seeds = pipeline.get("seeds", [config.get("seed", 0)])
        for variant in pipeline.get("compression_variants", []):
            if not (variant.get("export") and variant.get("benchmark")):
                continue
            variant_seeds = variant.get("seeds", seeds)
            for seed in variant.get("benchmark_seeds", [variant_seeds[0]]):
                deployment = experiment / "compression" / str(variant["name"]) / f"seed_{seed}" / "deployment"
                roots.append((deployment, targets))
    return roots


def _real_device_checks(
    results_root: Path,
    configs: list[tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected: list[Path] = []
    for deployment, targets in _deployment_roots(results_root, configs):
        for target in targets:
            variants = _REAL_DEVICE_PROTOCOL[target]
            target_name = target.replace("-", "_")
            for variant in variants:
                for repetition in (1, 2, 3):
                    expected.append(deployment / f"latency_{variant}_rep{repetition}_{target_name}.json")
    invalid_latency = []
    invalid_memory = []
    for path in expected:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid_latency.append(path)
            invalid_memory.append(path)
            continue
        if not isinstance(payload.get("latency_mean_ms"), (int, float)) or payload["latency_mean_ms"] <= 0:
            invalid_latency.append(path)
        memory_fields = ("peak_memory_mb", "process_peak_rss_mb", "process_incremental_peak_rss_mb")
        if any(not isinstance(payload.get(key), (int, float)) or payload[key] <= 0 for key in memory_fields):
            invalid_memory.append(path)
    return (
        _check_row("real_device_latency", expected, results_root, invalid_latency),
        _check_row("real_device_memory", expected, results_root, invalid_memory),
    )


def _completeness(
    results_root: Path,
    configs: list[tuple[Path, dict[str, Any]]],
) -> pd.DataFrame:
    expected = _expected_paths(results_root, configs)
    rows = [_teacher_gate_check(expected.pop("teacher_reproduction_gates"), results_root)]
    rows.extend(_check_row(name, paths, results_root) for name, paths in expected.items())
    rows.append(_compression_flop_check(expected["compression_summaries"], results_root))
    rows.extend(_real_device_checks(results_root, configs))
    return pd.DataFrame(rows)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _pytest_verification(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_recorded", "path": path.name}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return {"status": "invalid", "path": path.name}
    node = root if root.tag == "testsuite" else next(iter(root.findall("testsuite")), root)
    tests = int(root.attrib.get("tests", node.attrib.get("tests", 0)))
    failures = int(root.attrib.get("failures", node.attrib.get("failures", 0)))
    errors = int(root.attrib.get("errors", node.attrib.get("errors", 0)))
    skipped = int(root.attrib.get("skipped", node.attrib.get("skipped", 0)))
    return {
        "status": "passed" if tests > 0 and failures == 0 and errors == 0 else "failed",
        "path": path.name,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _privacy_audit(paths: Iterable[Path], project_root: Path) -> dict[str, Any]:
    scanned = 0
    leaks: list[str] = []
    text_suffixes = {".csv", ".json", ".md", ".yaml", ".yml", ".txt", ".py", ".sh", ".xml"}
    for path in sorted(set(paths)):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        scanned += 1
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        if _PRIVATE_PATH.search(content):
            leaks.append(_relative(path, project_root))
    return {"passed": not leaks, "scanned_files": scanned, "files_with_private_paths": leaks}


def freeze_results(results_dir: str | Path, *, strict_privacy: bool = True) -> dict[str, Any]:
    """Write integrity, coverage, provenance, and privacy records for final results."""

    project_root = Path(__file__).resolve().parents[2]
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    analysis = results_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    configs = _load_task_configs(project_root)

    completeness = _completeness(results_root, configs)
    completeness.to_csv(analysis / "result_completeness.csv", index=False)

    source_records = _file_records(_source_files(project_root), project_root)
    test_records = _file_records((project_root / "tests").glob("test_*.py"), project_root)
    result_paths = _result_files(results_root)
    result_records = _file_records(result_paths, project_root)
    pd.DataFrame(result_records).to_csv(analysis / "result_artifact_manifest.csv", index=False)

    privacy = _privacy_audit([*_source_files(project_root), *result_paths], project_root)
    if strict_privacy and not privacy["passed"]:
        raise RuntimeError(
            "Private absolute paths found in exported evidence: "
            + ", ".join(privacy["files_with_private_paths"])
        )

    config_records = _file_records((path for path, _ in configs), project_root)
    verification = _pytest_verification(analysis / "pytest.xml")
    artifacts_complete = bool(completeness["complete"].all())
    if not privacy["passed"]:
        status = "privacy_failed"
    elif not artifacts_complete:
        status = "incomplete"
    elif verification["status"] == "passed":
        status = "complete"
    else:
        status = f"complete_{verification['status']}"
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": _relative(results_root, project_root),
        "status": status,
        "completeness": {
            "all_required_complete": artifacts_complete,
            "checks": completeness.to_dict(orient="records"),
        },
        "integrity": {
            "source_tree_sha256": _aggregate_digest(source_records),
            "source_file_count": len(source_records),
            "test_tree_sha256": _aggregate_digest(test_records),
            "test_file_count": len(test_records),
            "result_artifacts_sha256": _aggregate_digest(result_records),
            "result_artifact_count": len(result_records),
            "task_configs": config_records,
        },
        "environment": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": {name: _package_version(name) for name in _CORE_PACKAGES},
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "git_commit": _git_commit(project_root),
        },
        "verification": verification,
        "privacy": privacy,
        "binary_artifacts": _binary_inventory(results_root, project_root),
    }
    (analysis / "result_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest
