"""Load canonical experiment summaries with task metadata and teacher baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_EXCLUDED_PARTS = {"cache", "analysis", "warmup", "common_warmup", "operating_points"}


def _config_for(summary_path: Path) -> dict:
    path = summary_path.parent / "config.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _enrich(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = frame.copy()
    task = str(config.get("task_type", out.get("task", pd.Series(["task"])).iloc[0]))
    out["task"] = task
    out["dataset"] = str(config.get("dataset", {}).get("name", "unknown"))
    model_role = str(config.get("evaluation", {}).get("model_role") or "compressed_model")
    out["architecture"] = str(config.get(model_role, {}).get("name", "unknown"))
    if task == "classification" and "auc" in out:
        auc = pd.to_numeric(out["auc"], errors="coerce")
        invalid_auc = ~np.isfinite(auc)
        out["score"] = auc.mask(invalid_auc, 0.5)
        out["score_status"] = np.where(
            invalid_auc,
            "undefined_auc_imputed_chance",
            "observed_auc",
        )
    elif task.startswith("segmentation") and "dice" in out:
        out["score"] = pd.to_numeric(out["dice"], errors="coerce")
        out["score_status"] = "observed_dice"
    return out


def _deployment_metrics(summary_path: Path) -> dict:
    """Load measured deployment fields colocated with an experiment summary."""

    result_dir=summary_path.parent
    if result_dir.parent.name=="results" and result_dir.parent.parent.name=="deployment":
        path=result_dir.parent.parent/f"latency_{result_dir.name}.json"
    else:
        path = result_dir / "deployment" / "latency.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        key: payload[key]
        for key in ("latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p95_ms", "throughput_fps", "peak_memory_mb")
        if key in payload
    }


_REFERENCE_METRICS = (
    "score",
    "ece",
    "false_negative_lesion_rate",
    "small_component_recall",
    "boundary_f1",
)


def _summary_metrics(path: Path, task: str) -> dict[str, float]:
    """Read performance and reliability metrics from a source experiment."""

    summary=path/"summary.csv"
    if not summary.exists():return {}
    frame=pd.read_csv(summary)
    column="auc" if task=="classification" else "dice"
    metrics: dict[str, float] = {}
    for output_name, input_name in (("score", column), *((name, name) for name in _REFERENCE_METRICS[1:])):
        if input_name not in frame:
            continue
        values=pd.to_numeric(frame[input_name],errors="coerce").dropna()
        if not values.empty:
            metrics[output_name]=float(values.iloc[0])
    return metrics


def _declared_compression_variant(summary_path: Path, config: dict) -> dict | None:
    """Return the pipeline variant matching a compression result directory.

    Older resumable runs may predate the persisted ``reporting`` metadata even
    though their saved config still contains the pipeline declaration.  The
    directory name and declaration together are an unambiguous provenance
    source; a directory named ``compressed`` alone is deliberately insufficient
    because users may choose that word for an ordinary experiment.
    """

    result_dir=summary_path.parent
    if result_dir.parent.parent.name not in {"compression","compressed"}:
        return None
    variant_name=result_dir.parent.name
    for variant in config.get("pipeline",{}).get("compression_variants",[]):
        if str(variant.get("name"))==variant_name:
            return variant
    # Some early resumable artifacts were written before the variant was added
    # to the saved pipeline config. A persisted compression manifest is direct
    # evidence of the stage that produced this result and is safer than
    # inferring provenance from a directory name alone.
    if (result_dir/"compression"/"compression_metrics.json").exists():
        return {}
    return None


def _inferred_source_dir(
    summary_path:Path,frame:pd.DataFrame,source_method:str
) -> Path | None:
    """Infer a trained compression source colocated with a declared variant."""

    if source_method=="teacher":return None
    compressed_dir=summary_path.parent.parent.parent
    experiment_root=compressed_dir.parent
    seed=(
        int(pd.to_numeric(frame["seed"],errors="coerce").dropna().iloc[0])
        if "seed" in frame and not pd.to_numeric(frame["seed"],errors="coerce").dropna().empty
        else 0
    )
    category="training" if compressed_dir.name=="compression" else None
    candidate=(experiment_root/category/source_method/f"seed_{seed}" if category
               else experiment_root/source_method/f"seed_{seed}")
    return candidate if (candidate/"summary.csv").exists() else None


def load_canonical_results(results_dir: str | Path, include_teachers: bool = True) -> pd.DataFrame:
    """Load result summaries while excluding diagnostics and enriching metadata."""

    root = Path(results_dir)
    frames: list[pd.DataFrame] = []
    teacher_specs: dict[tuple[str, str], tuple[Path, dict]] = {}
    for path in sorted(root.rglob("summary.csv")):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        config = _config_for(path)
        frame = _enrich(pd.read_csv(path), config)
        frame["result_dir"] = str(path.parent)
        frame["is_comparison_reference"] = False
        reporting=config.get("reporting",{})
        declared_variant=_declared_compression_variant(path,config)
        if "deployment" in path.parts and "results" in path.parts:
            frame["comparison_family"]="quantization"
            frame["reference_method"]="onnx_fp32"
            frame["comparison_group"]=str(config.get("output_dir",path.parent.parent.parent))
        elif reporting.get("comparison_family")=="pruning" or declared_variant is not None:
            frame["comparison_family"]="pruning"
            source_method=str(
                reporting.get("compression_source_method")
                or (declared_variant or {}).get("method","racmed")
            )
            frame["reference_method"]=source_method
            frame["comparison_group"]=str(config.get("output_dir",path.parent))
            source_dir=reporting.get("compression_source_result_dir")
            inferred=_inferred_source_dir(path,frame,source_method) if not source_dir else None
            reference_path=Path(source_dir) if source_dir else inferred
            reference_metrics=(
                _summary_metrics(reference_path,str(frame["task"].iloc[0]))
                if reference_path is not None else {}
            )
            for metric,value in reference_metrics.items():
                frame[f"reference_{metric}"]=value
        else:
            frame["comparison_family"]="knowledge_distillation"
            frame["reference_method"]="teacher"
            frame["comparison_group"]=str(frame["task"].iloc[0])+"::"+str(frame["dataset"].iloc[0])
        for key, value in _deployment_metrics(path).items():
            frame[key] = value
        frames.append(frame)
        teacher_dir = config.get("comparison_teacher_output_dir", config.get("teacher_output_dir"))
        if teacher_dir:
            key = (str(frame["task"].iloc[0]), str(frame["dataset"].iloc[0]))
            teacher_specs[key] = (Path(teacher_dir), config)
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True)
    if not include_teachers:
        return results
    teacher_rows: list[pd.DataFrame] = []
    for (task, dataset), (directory, config) in teacher_specs.items():
        summary = directory / "summary.csv"
        if not summary.exists():
            continue
        teacher = _enrich(pd.read_csv(summary), config)
        teacher["result_dir"] = str(directory)
        teacher["method"] = "teacher"
        teacher["architecture"] = str(config.get("teacher", {}).get("name", "teacher"))
        seeds = sorted(pd.to_numeric(results.loc[results["dataset"] == dataset, "seed"], errors="coerce").dropna().astype(int).unique())
        for seed in seeds or [0]:
            row = teacher.copy()
            row["seed"] = seed
            row["is_comparison_reference"] = True
            teacher_rows.append(row)
    if teacher_rows:
        results = pd.concat([results, *teacher_rows], ignore_index=True)
        reference_units = {
            (str(row["task"]), str(row["dataset"]))
            for _, row in results.loc[
                results["is_comparison_reference"].fillna(False).astype(bool)
            ].iterrows()
        }
        redundant_teacher = results.apply(
            lambda row: (
                str(row.get("method")) == "teacher"
                and not bool(row.get("is_comparison_reference", False))
                and (str(row.get("task")), str(row.get("dataset"))) in reference_units
            ),
            axis=1,
        )
        results = results.loc[~redundant_teacher].reset_index(drop=True)
    for metric in _REFERENCE_METRICS:
        reference_column=f"reference_{metric}"
        results[reference_column]=pd.to_numeric(results.get(reference_column),errors="coerce")
    comparison_reference = results[
        results.method.astype(str).eq("teacher")
        & results["is_comparison_reference"].fillna(False).astype(bool)
    ]
    if comparison_reference.empty:
        comparison_reference = results.loc[results.method.astype(str).eq("teacher")]
    reference_tables: dict[tuple[str, str], pd.Series] = {}
    for metric in _REFERENCE_METRICS:
        if metric not in results:
            continue
        reference_tables[("teacher",metric)]=(comparison_reference
            .groupby(["task","dataset","seed"],dropna=False)[metric].mean())
        reference_tables[("onnx_fp32",metric)]=(results.loc[results.method.astype(str).eq("onnx_fp32")]
            .groupby(["comparison_group","seed"],dropna=False)[metric].mean())
    for index,row in results.iterrows():
        family=row.get("comparison_family")
        for metric in _REFERENCE_METRICS:
            column=f"reference_{metric}"
            if pd.notna(row.get(column)):
                continue
            if family=="quantization":
                table=reference_tables.get(("onnx_fp32",metric))
                key=(row.get("comparison_group"),row.get("seed"))
            elif (
                family=="knowledge_distillation"
                and str(row.get("method"))!="teacher"
            ) or (
                family=="pruning"
                and str(row.get("reference_method"))=="teacher"
            ):
                table=reference_tables.get(("teacher",metric))
                key=(row.get("task"),row.get("dataset"),row.get("seed"))
            else:
                continue
            if table is not None and key in table.index and pd.notna(table.loc[key]):
                results.at[index,column]=float(table.loc[key])
    return results
