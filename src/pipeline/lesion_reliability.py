"""Inference stage for threshold-independent lesion reliability evaluation."""

from __future__ import annotations

from itertools import islice
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.datasets.dataset_factory import create_dataloaders
from src.engine.checkpoint import load_model_checkpoint
from src.metrics.lesion_operating_curve import LesionOperatingCurve
from src.metrics.statistical_tests import bootstrap_ci,paired_tests
from src.models.forward import forward_model
from src.models.model_factory import create_model
from src.pipeline.model_artifacts import pipeline_checkpoint,pipeline_model_role
from src.pipeline.progress import pipeline_logger
from src.utils.progress import progress_bar


@torch.inference_mode()
def evaluate_lesion_operating_curve(
    config:dict[str,Any],checkpoint:str|Path|None=None,method_label:str|None=None,
    model_role:str|None=None,debug:bool=False,
) -> dict[str,Any]:
    """Evaluate FROC and recall at matched false-positive budgets on the test split."""

    if not str(config.get("task_type","")).startswith("segmentation"):
        raise ValueError("Lesion operating curves require a segmentation task.")
    role=model_role or pipeline_model_role(config,"evaluation")
    model=create_model(config,role)
    selected=pipeline_checkpoint(config,config.get("output_dir","outputs/experiment"),"evaluation",checkpoint)
    if selected is not None:
        if not selected.exists():raise FileNotFoundError(f"Evaluation checkpoint not found: {selected}")
        model,_=load_model_checkpoint(selected,model)
    device=torch.device("cuda" if torch.cuda.is_available() and config.get("device","auto")!="cpu" else "cpu")
    model=model.to(device).eval();loader=create_dataloaders(config,debug)["test"]
    evaluation=config.get("evaluation",{});training=config.get("training",{})
    thresholds=tuple(evaluation.get("lesion_threshold_candidates",np.linspace(.05,.95,19).tolist()+[.99,1.0]))
    small_threshold=int(evaluation.get("small_component_threshold",config.get("risk",{}).get("small_component_threshold",64)))
    curve=LesionOperatingCurve(
        thresholds,small_threshold,
        float(evaluation.get("lesion_component_min_iou",.10)),
        float(evaluation.get("froc_max_fp_per_case",4.0)),
        int(evaluation.get("froc_workers",1)),
    )
    sliding=evaluation.get("sliding_window")
    max_batches=training.get("max_test_batches",training.get("max_eval_batches"))
    total=min(len(loader),int(max_batches)) if max_batches is not None else len(loader)
    source=islice(loader,total) if max_batches is not None else loader
    current_group=None;group_logits=[];group_targets=[]
    def flush() -> None:
        nonlocal group_logits,group_targets
        if not group_logits:return
        if group_logits[0].ndim==3:
            logits=torch.stack(group_logits,dim=1).unsqueeze(0)
            stack_dim=1 if group_targets[0].ndim==group_logits[0].ndim and group_targets[0].shape[0]==group_logits[0].shape[0] else 0
            targets=torch.stack(group_targets,dim=stack_dim).unsqueeze(0)
        else:
            logits=group_logits[0].unsqueeze(0);targets=group_targets[0].unsqueeze(0)
        curve.update(logits,targets);group_logits=[];group_targets=[]
    for batch in progress_bar(source,desc=f"Lesion FROC {method_label or role}",total=total,unit="batch",leave=True):
        image=batch["image"].to(device,non_blocking=device.type=="cuda")
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
            if sliding:
                from monai.inferers import sliding_window_inference
                logits=sliding_window_inference(
                    image,roi_size=tuple(sliding.get("roi_size",[96,96,96])),
                    sw_batch_size=int(sliding.get("sw_batch_size",1)),
                    predictor=lambda window:model(window,return_features=False)["logits"],
                    overlap=float(sliding.get("overlap",.25)),
                )
            else:logits=forward_model(model,image,batch,False)["logits"]
        group_ids=batch.get("group_id",batch["id"]);targets=batch["target"]
        for index,group_id in enumerate(group_ids):
            group_id=str(group_id)
            if current_group is not None and group_id!=current_group:flush()
            current_group=group_id;group_logits.append(logits[index].float().cpu());group_targets.append(targets[index].cpu())
    flush();summary,rows=curve.compute(
        evaluation.get("froc_fp_targets",(.25,.5,1.,2.,4.)),
        evaluation.get("froc_auc_fp_limits"),
    )
    summary.update({
        "method":method_label or role,"seed":int(config.get("seed",0)),
        "task":str(config.get("task_type")),"dataset":str(config.get("dataset",{}).get("name","unknown")),
        "evaluation_split":"test","threshold_selection":"threshold-independent_test_curve",
    })
    output=Path(config.get("output_dir","outputs/experiment"));output.mkdir(parents=True,exist_ok=True)
    (output/"lesion_operating_curve.json").write_text(json.dumps({"summary":summary,"curve":rows},indent=2))
    pd.DataFrame(rows).to_csv(output/"lesion_operating_curve.csv",index=False)
    pd.DataFrame([summary]).to_csv(output/"lesion_operating_curve_summary.csv",index=False)
    pipeline_logger().info(
        "LESION_FROC_RESULT | method=%s seed=%d auc=%.4f small_auc=%.4f recall_at_1fp=%.4f",
        summary["method"],summary["seed"],
        summary[summary["primary_froc_auc_key"]],
        summary["small_"+summary["primary_froc_auc_key"]],
        summary["lesion_recall_at_1_fp_per_case"],
    )
    return summary


def aggregate_lesion_operating_curves(results_dir:str|Path) -> pd.DataFrame:
    """Aggregate seed-level FROC outputs without mixing dataset identities."""

    root=Path(results_dir);frames=[]
    for path in sorted(root.rglob("lesion_operating_curve_summary.csv")):
        frame=pd.read_csv(path);frame["result_dir"]=path.parent.relative_to(root).as_posix();frames.append(frame)
    analysis=root/"analysis";analysis.mkdir(parents=True,exist_ok=True)
    if not frames:
        empty=pd.DataFrame();empty.to_csv(analysis/"lesion_operating_curve_statistics.csv",index=False)
        return empty
    records=pd.concat(frames,ignore_index=True)
    records.to_csv(analysis/"lesion_operating_curve_records.csv",index=False)
    metrics=[column for column in records if column.startswith(("froc_auc_","small_froc_auc_","lesion_recall_at_","small_lesion_recall_at_"))]
    rows=[]
    for keys,subset in records.groupby(["task","dataset","method"],dropna=False):
        for metric in metrics:
            values=pd.to_numeric(subset[metric],errors="coerce").dropna().to_numpy()
            if not len(values):continue
            low,high=bootstrap_ci(values,n_resamples=2000,seed=0)
            rows.append({
                "task":keys[0],"dataset":keys[1],"method":keys[2],"metric":metric,
                "mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else 0.0,
                "count":len(values),"ci95_low":low,"ci95_high":high,
            })
    statistics=pd.DataFrame(rows);statistics.to_csv(analysis/"lesion_operating_curve_statistics.csv",index=False)
    comparisons=[]
    for (task,dataset),subset in records.groupby(["task","dataset"],dropna=False):
        for reference in ("standalone","vanilla_kd"):
            left=subset[subset.method.eq("racmed")];right=subset[subset.method.eq(reference)]
            paired=left.merge(right,on="seed",suffixes=("_racmed","_reference"))
            for metric in metrics:
                a=pd.to_numeric(paired.get(f"{metric}_racmed"),errors="coerce")
                b=pd.to_numeric(paired.get(f"{metric}_reference"),errors="coerce")
                valid=a.notna()&b.notna()
                if not valid.any():continue
                differences=(a[valid]-b[valid]).to_numpy();low,high=bootstrap_ci(differences,n_resamples=2000,seed=0)
                tests=paired_tests(a[valid],b[valid]) if valid.sum()>=2 else {"wilcoxon_p":float("nan"),"paired_t_p":float("nan")}
                comparisons.append({
                    "task":task,"dataset":dataset,"reference_method":reference,"metric":metric,
                    "n_pairs":int(valid.sum()),"mean_difference":float(differences.mean()),
                    "ci95_low":low,"ci95_high":high,**tests,
                })
    pd.DataFrame(comparisons).to_csv(analysis/"lesion_operating_curve_paired_tests.csv",index=False)
    return statistics
