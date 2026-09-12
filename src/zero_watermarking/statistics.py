from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ci(values, statistic=np.mean, confidence=0.95, n_boot=5000, seed=42):
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, x.size, size=(n_boot, x.size))
    estimates = np.asarray([statistic(x[idx]) for idx in samples])
    alpha = (1 - confidence) / 2
    return float(statistic(x)), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5: return {"statistic": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    stat, p = wilcoxon(a[mask], b[mask], zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "n": int(mask.sum())}


def summarize_repeated_runs(frame: pd.DataFrame, group_cols=("method", "attack"), metric="nc"):
    rows = []
    for keys, g in frame.groupby(list(group_cols)):
        mean, lo, hi = bootstrap_ci(g[metric].to_numpy())
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)
