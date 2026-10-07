"""Sensitivity analyses for contradiction thresholds and RAS penalties."""

from __future__ import annotations

import pandas as pd


def contradiction_threshold_sensitivity(
    drop_matrix: pd.DataFrame,
    epsilons: tuple[float, ...] = (.01, .02, .03),
    taus: tuple[float, ...] = (.05, .06, .07, .08, .09, .10),
) -> pd.DataFrame:
    """Evaluate contradiction labels over a predeclared threshold grid."""

    rows=[]
    for epsilon in epsilons:
        for tau in taus:
            for method,values in drop_matrix.iterrows():
                valid=pd.to_numeric(values,errors="coerce").dropna()
                safe=int((valid<=epsilon).sum())
                harmful=int((valid>=tau).sum())
                rows.append({"method":method,"epsilon":epsilon,"tau":tau,"safe_tasks":safe,"harmful_tasks":harmful,"is_contradiction":bool(safe>=2 and harmful>=1)})
    return pd.DataFrame(rows)


def ras_weight_sensitivity(common_ras: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Re-rank common-coverage methods under predeclared penalty scenarios."""

    scenarios={
        "default":(1.,1.,1.,.1),
        "equal_no_latency":(1.,1.,1.,0.),
        "clinical_reliability":(1.,.5,1.5,0.),
        "performance_centered":(.5,.5,.5,.1),
        "latency_emphasized":(1.,1.,1.,.5),
    }
    rows=[]
    for scenario,(alpha,beta,gamma,delta) in scenarios.items():
        for row in common_ras.itertuples(index=False):
            score=float(row.average_performance-alpha*row.worst_task_drop-beta*row.ece-gamma*row.false_negative_rate-delta*row.normalized_latency)
            rows.append({"scenario":scenario,"method":row.method,"alpha":alpha,"beta":beta,"gamma":gamma,"delta":delta,"risk_adjusted_score":score})
    columns=["scenario","method","alpha","beta","gamma","delta","risk_adjusted_score","rank"]
    if not rows:
        return pd.DataFrame(columns=columns),pd.DataFrame(columns=["method","mean_rank","best_rank","worst_rank","top_rank_count"])
    frame=pd.DataFrame(rows)
    frame["rank"]=frame.groupby("scenario")["risk_adjusted_score"].rank(method="min",ascending=False).astype(int)
    stability=frame.groupby("method",as_index=False).agg(mean_rank=("rank","mean"),best_rank=("rank","min"),worst_rank=("rank","max"),top_rank_count=("rank",lambda values:int((values==1).sum())))
    return frame,stability
