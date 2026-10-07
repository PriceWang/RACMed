"""Bootstrap intervals and paired significance tests."""

from __future__ import annotations
import numpy as np
from scipy import stats


def bootstrap_ci(values, confidence: float = .95, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_resamples)])
    q = (1-confidence)/2
    return float(np.quantile(means,q)), float(np.quantile(means,1-q))


def paired_tests(a, b) -> dict[str, float]:
    a, b = np.asarray(a), np.asarray(b)
    try: wilcoxon = float(stats.wilcoxon(a,b).pvalue)
    except ValueError: wilcoxon = 1.0
    return {"wilcoxon_p": wilcoxon, "paired_t_p": float(stats.ttest_rel(a,b).pvalue)}
