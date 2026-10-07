"""Grouped contradiction detection and statistical result export."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from .risk_adjusted_score import risk_adjusted_score
from .sensitivity import contradiction_threshold_sensitivity,ras_weight_sensitivity
from src.metrics.statistical_tests import bootstrap_ci, paired_tests
from src.utils.method_naming import normalize_method_names


def _mean_or_zero(frame: pd.DataFrame, *columns: str) -> float:
    """Return the first available non-NaN column mean, otherwise zero."""

    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return 0.0


def _add_evaluation_unit(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["evaluation_unit"] = out["task"].astype(str)
    if "dataset" in out:
        out["evaluation_unit"] += "::" + out["dataset"].astype(str)
    return out


def _normalize_latency(frame: pd.DataFrame, teacher_mask: pd.Series) -> pd.DataFrame:
    """Normalize latency as relative slowdown from the fastest compressed method.

    A min-max transform maps even sub-millisecond benchmark jitter to the full
    ``[0, 1]`` range when methods have similar runtimes. Relative slowdown keeps
    the penalty proportional to the observed deployment cost and clips only
    slowdowns above 100 percent.
    """

    out = frame.copy()
    if "latency_mean_ms" not in out:
        out["normalized_latency"] = float("nan")
        return out
    out["latency_mean_ms"] = pd.to_numeric(out["latency_mean_ms"], errors="coerce")
    out["normalized_latency"] = float("nan")
    groups = ["dataset"] if "dataset" in out else ["task"]
    group_key=groups[0] if len(groups)==1 else groups
    for _, indices in out.loc[~teacher_mask].groupby(group_key, dropna=False).groups.items():
        values = out.loc[indices, "latency_mean_ms"]
        valid = values.dropna()
        if valid.empty:
            continue
        fastest = float(valid.min())
        if fastest <= 0:
            out.loc[valid.index, "normalized_latency"] = 0.0
            continue
        out.loc[valid.index, "normalized_latency"] = ((valid / fastest) - 1.0).clip(0.0, 1.0)
    return out


def _ras_row(method: str, subset: pd.DataFrame, worst_drop: float, penalties: tuple[float, ...]) -> dict:
    average = float(subset.score.mean())
    ece = _mean_or_zero(subset, "ece")
    fnr = _mean_or_zero(subset, "false_negative_rate", "false_negative_lesion_rate")
    latency = _mean_or_zero(subset, "normalized_latency")
    latency_count = int(subset.loc[pd.to_numeric(subset.get("latency_mean_ms"), errors="coerce").notna(), "evaluation_unit"].nunique()) if "latency_mean_ms" in subset else 0
    return {
        "method": method,
        "average_performance": average,
        "worst_task_drop": worst_drop,
        "ece": ece,
        "false_negative_rate": fnr,
        "normalized_latency": latency,
        "latency_coverage_count": latency_count,
        "risk_adjusted_score": risk_adjusted_score(average, worst_drop, ece, fnr, latency, *penalties),
    }


def _paired_case_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap paired per-case score effects without pooling task identities."""

    if "result_dir" not in frame or "seed" not in frame:
        return pd.DataFrame()
    rows=[]
    pairs=(("racmed","standalone_compressed"),("racmed","vanilla_kd"))
    for keys,subset in frame.groupby(["task","dataset"],dropna=False):
        task,dataset=keys
        for reference,comparison in pairs:
            seed_means=[]
            for seed in sorted(pd.to_numeric(subset.seed,errors="coerce").dropna().astype(int).unique()):
                selected={}
                for method in (reference,comparison):
                    candidates=subset[(subset.method==method)&(pd.to_numeric(subset.seed,errors="coerce")==seed)]
                    if candidates.empty:continue
                    path=Path(str(candidates.iloc[0].result_dir))/"per_case_metrics.csv"
                    if path.exists():selected[method]=pd.read_csv(path)[["case_id","score"]].rename(columns={"score":method})
                if len(selected)!=2:continue
                paired=selected[reference].merge(selected[comparison],on="case_id")
                differences=(paired[reference]-paired[comparison]).to_numpy()
                if not len(differences):continue
                low,high=bootstrap_ci(differences,n_resamples=2000,seed=seed)
                mean=float(differences.mean());seed_means.append(mean)
                rows.append({"task":task,"dataset":dataset,"reference_method":reference,"comparison_method":comparison,"level":"case","seed":seed,"n_pairs":len(differences),"mean_difference":mean,"ci95_low":low,"ci95_high":high})
            if seed_means:
                low,high=bootstrap_ci(seed_means,n_resamples=2000,seed=0)
                rows.append({"task":task,"dataset":dataset,"reference_method":reference,"comparison_method":comparison,"level":"seed","seed":"all","n_pairs":len(seed_means),"mean_difference":float(pd.Series(seed_means).mean()),"ci95_low":low,"ci95_high":high})
    return pd.DataFrame(rows)


_RELIABILITY_DIRECTIONS = {
    # Positive values always mean that compression made reliability worse.
    "ece": 1.0,
    "false_negative_lesion_rate": 1.0,
    "small_component_recall": -1.0,
    "boundary_f1": -1.0,
}


def _reliability_contradictions(
    frame: pd.DataFrame,
    directory: Path,
    epsilon: float,
    tau: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export pre-specified reliability drops and cross-task contradictions."""

    records: list[pd.DataFrame] = []
    flags: list[dict] = []
    for metric, direction in _RELIABILITY_DIRECTIONS.items():
        reference=f"reference_{metric}"
        if metric not in frame or reference not in frame:
            continue
        subset=frame.copy()
        subset[metric]=pd.to_numeric(subset[metric],errors="coerce")
        subset[reference]=pd.to_numeric(subset[reference],errors="coerce")
        subset=subset.dropna(subset=[metric,reference])
        if subset.empty:
            continue
        subset["reliability_metric"]=metric
        subset["reference_value"]=subset[reference]
        subset["compressed_value"]=subset[metric]
        subset["reliability_drop"]=direction*(subset[metric]-subset[reference])
        group=["method","task"]+(["dataset"] if "dataset" in subset else [])+(["comparison_family"] if "comparison_family" in subset else [])
        grouped=subset.groupby(group,dropna=False,as_index=False).agg(
            reference_value=("reference_value","mean"),
            compressed_value=("compressed_value","mean"),
            reliability_drop=("reliability_drop","mean"),
        )
        grouped["reliability_metric"]=metric
        grouped=_add_evaluation_unit(grouped)
        records.append(grouped)
        matrix=grouped.pivot_table(index="method",columns="evaluation_unit",values="reliability_drop",aggfunc="mean").sort_index()
        matrix.to_csv(directory/f"contradiction_matrix_reliability_{metric}.csv")
        for method,row in matrix.iterrows():
            valid=row.dropna()
            safe=int((valid<=epsilon).sum())
            harmful=int((valid>=tau).sum())
            flags.append({
                "method":method,
                "reliability_metric":metric,
                "safe_tasks":safe,
                "harmful_tasks":harmful,
                "worst_reliability_drop":float(valid.max()) if len(valid) else float("nan"),
                "is_reliability_contradiction":bool(safe>=2 and harmful>=1),
            })
    record_frame=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    flag_frame=pd.DataFrame(flags)
    record_frame.to_csv(directory/"reliability_drop_records.csv",index=False)
    flag_frame.to_csv(directory/"reliability_contradictions.csv",index=False)
    return record_frame,flag_frame


class CompressionContradictionAnalyzer:
    """Flag methods that preserve at least two tasks but fail materially on another."""
    def __init__(self,epsilon=.01,tau=.05,alpha=1.,beta=1.,gamma=1.,delta=.1):
        self.epsilon,self.tau=epsilon,tau;self.penalties=(alpha,beta,gamma,delta)
    def analyze(self,frame:pd.DataFrame,output_dir:str|Path):
        frame=normalize_method_names(frame)
        required={"method","task","score"};missing=required-set(frame)
        if missing:raise ValueError(f"Results missing columns: {sorted(missing)}")
        directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
        method_names=frame.method.astype(str)
        teacher_mask=method_names.eq("teacher") | method_names.str.startswith("teacher_")
        frame=_add_evaluation_unit(_normalize_latency(frame,teacher_mask))
        method_names=frame.method.astype(str)
        teacher_mask=method_names.eq("teacher") | method_names.str.startswith("teacher_")
        group=[c for c in ("method","task","dataset","architecture") if c in frame]
        means=frame.groupby(group,dropna=False)["score"].agg(["mean","std","count"]).reset_index()
        intervals=[]
        for keys,subset in frame.groupby(group,dropna=False):
            low,high=bootstrap_ci(subset.score.to_numpy(),n_resamples=500)
            key_values=(keys,) if not isinstance(keys,tuple) else keys
            intervals.append({**dict(zip(group,key_values)),"ci95_low":low,"ci95_high":high})
        means=means.merge(pd.DataFrame(intervals),on=group,how="left");means["mean_std"]=means.apply(lambda row:f"{row['mean']:.4f} ± {row['std']:.4f}" if pd.notna(row["std"]) else f"{row['mean']:.4f} ± 0.0000",axis=1)
        means.to_csv(directory/"grouped_statistics.csv",index=False)
        units=["task"]+(["dataset"] if "dataset" in frame else [])
        comparable=frame[~teacher_mask].copy()
        if "reference_score" in comparable and pd.to_numeric(comparable["reference_score"],errors="coerce").notna().any():
            comparable["reference_score"]=pd.to_numeric(comparable["reference_score"],errors="coerce")
            comparable["drop"]=comparable.reference_score-comparable.score
            comparison_group=["method",*units]+(["comparison_family"] if "comparison_family" in comparable else [])
            methods=comparable.groupby(comparison_group,dropna=False,as_index=False).agg(score=("score","mean"),reference_score=("reference_score","mean"),drop=("drop","mean"))
        else:
            teachers=frame[teacher_mask].groupby(units,dropna=False,as_index=False)["score"].max().rename(columns={"score":"reference_score"})
            methods=comparable.groupby(["method",*units],dropna=False,as_index=False)["score"].mean()
            methods=methods.merge(teachers,on=units,how="left");methods["drop"]=methods.reference_score-methods.score
        methods=_add_evaluation_unit(methods)
        methods.to_csv(directory/"comparison_drop_records.csv",index=False)
        matrix=methods.pivot(index="method",columns="evaluation_unit",values="drop").sort_index()
        matrix.to_csv(directory/"contradiction_matrix.csv")
        if "comparison_family" in methods:
            for family,family_frame in methods.groupby("comparison_family",dropna=False):
                family_matrix=family_frame.pivot(index="method",columns="evaluation_unit",values="drop").sort_index()
                family_matrix.to_csv(directory/f"contradiction_matrix_{str(family)}.csv")
        coverage_rows=[]
        for method,row in matrix.iterrows():
            units_present=sorted(row.dropna().index.astype(str))
            coverage_rows.append({"method":method,"coverage_count":len(units_present),"coverage_signature":"|".join(units_present)})
        coverage=pd.DataFrame(coverage_rows)
        coverage.to_csv(directory/"method_coverage.csv",index=False)
        contradictions=[];worst=[];ras=[]
        for method,row in matrix.iterrows():
            valid=row.dropna();safe=int((valid<=self.epsilon).sum());harmful=int((valid>=self.tau).sum());worst_drop=float(valid.max()) if len(valid) else 0.
            family_values=methods.loc[methods.method==method,"comparison_family"].dropna().astype(str).unique() if "comparison_family" in methods else []
            family="|".join(sorted(family_values)) if len(family_values) else "unspecified"
            contradictions.append({"method":method,"comparison_family":family,"safe_tasks":safe,"harmful_tasks":harmful,"is_contradiction":bool(safe>=2 and harmful>=1)})
            worst_task=valid.idxmax() if len(valid) else None;worst.append({"method":method,"worst_task":worst_task,"worst_task_drop":worst_drop})
            subset=frame[frame.method==method]
            ras.append({**_ras_row(method,subset,worst_drop,self.penalties),"coverage_count":len(valid),"coverage_signature":"|".join(sorted(valid.index.astype(str)))})
        cdf=pd.DataFrame(contradictions);wdf=pd.DataFrame(worst);rdf=pd.DataFrame(ras)
        cdf.to_csv(directory/"contradiction_flags.csv",index=False)
        wdf.to_csv(directory/"worst_task_drop.csv",index=False);rdf.to_csv(directory/"risk_adjusted_score.csv",index=False)
        common_rows=[]
        if not coverage.empty:
            threshold=max(1,int(coverage.coverage_count.max())-1)
            eligible=coverage.loc[coverage.coverage_count>=threshold,"method"].tolist()
            common_units=set(matrix.columns)
            for method in eligible:
                common_units &= set(matrix.loc[method].dropna().index)
            for method in eligible:
                subset=frame[(frame.method==method)&frame.evaluation_unit.isin(common_units)]
                drops=matrix.loc[method,list(common_units)].dropna() if common_units else pd.Series(dtype=float)
                common_rows.append({**_ras_row(method,subset,float(drops.max()) if len(drops) else 0.,self.penalties),"common_coverage_count":len(common_units),"common_coverage_signature":"|".join(sorted(common_units))})
        common_ras=pd.DataFrame(common_rows)
        common_ras.to_csv(directory/"common_task_risk_adjusted_score.csv",index=False)
        threshold_sensitivity=contradiction_threshold_sensitivity(matrix)
        threshold_sensitivity.to_csv(directory/"contradiction_threshold_sensitivity.csv",index=False)
        ras_sensitivity,ras_stability=ras_weight_sensitivity(common_ras)
        ras_sensitivity.to_csv(directory/"ras_weight_sensitivity.csv",index=False)
        ras_stability.to_csv(directory/"ras_rank_stability.csv",index=False)
        test_rows=[]
        if "seed" in frame:
            strata=[c for c in ("task","dataset") if c in frame]
            grouped=frame.groupby(strata,dropna=False) if strata else [((),frame)]
            for keys,unit_frame in grouped:
                key_values=keys if isinstance(keys,tuple) else (keys,)
                unit=dict(zip(strata,key_values))
                aggregated=unit_frame.groupby(["method","seed"],dropna=False,as_index=False)["score"].mean()
                names=sorted(
                    aggregated.method.astype(str).unique(),
                    key=lambda name:(not (name=="teacher" or name.startswith("teacher_")),name),
                )
                for index,reference in enumerate(names):
                    for comparison in names[index+1:]:
                        left=aggregated[aggregated.method==reference]
                        right=aggregated[aggregated.method==comparison]
                        paired=left.merge(right,on=["seed"],suffixes=("_reference","_comparison"))
                        if len(paired)>=2:
                            test_rows.append({
                                **unit,
                                "method":comparison if reference=="teacher" else f"{reference}_vs_{comparison}",
                                "reference_method":reference,
                                "comparison_method":comparison,
                                "n_pairs":len(paired),
                                **paired_tests(paired.score_reference,paired.score_comparison),
                            })
        pd.DataFrame(test_rows,columns=[*(["task"] if "task" in frame else []),*(["dataset"] if "dataset" in frame else []),"method","reference_method","comparison_method","n_pairs","wilcoxon_p","paired_t_p"]).to_csv(directory/"statistical_tests.csv",index=False)
        case_effects=_paired_case_effects(frame)
        case_effects.to_csv(directory/"paired_per_case_bootstrap.csv",index=False)
        reliability_drops,reliability_flags=_reliability_contradictions(frame,directory,self.epsilon,self.tau)
        return {"contradictions":cdf,"matrix":matrix,"worst_task_drop":wdf,"risk_adjusted_score":rdf,"common_task_risk_adjusted_score":common_ras,"coverage":coverage,"grouped_statistics":means,"contradiction_threshold_sensitivity":threshold_sensitivity,"ras_weight_sensitivity":ras_sensitivity,"ras_rank_stability":ras_stability,"paired_per_case_bootstrap":case_effects,"reliability_drop_records":reliability_drops,"reliability_contradictions":reliability_flags}
