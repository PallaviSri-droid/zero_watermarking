from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ci95(values: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if x.size == 1:
        return mean, float("nan"), float("nan")
    # Small-seed experiments use the Student-t interval; this avoids pretending
    # that five seeds justify a normal approximation.
    from scipy.stats import t
    half = float(t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))
    return mean, mean - half, mean + half


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate CAP-ZW seed-level evaluator summaries.")
    parser.add_argument("inputs", nargs="+", help="Seed summary.csv files or directories containing summary.csv")
    parser.add_argument("--out", default="experiments/results/multi_seed_summary.csv")
    args = parser.parse_args()

    rows = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            path = path / "summary.csv"
        if not path.exists():
            raise SystemExit(f"Missing summary: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["source"] = str(path)
        rows.append(row)
    if not rows:
        raise SystemExit("No non-empty summary files were supplied.")

    raw = pd.DataFrame(rows)
    numeric = [c for c in raw.columns if c not in {"version", "split", "source"} and pd.api.types.is_numeric_dtype(raw[c])]
    group_cols = [c for c in ("version", "split", "bits") if c in raw.columns]
    summaries = []
    for keys, group in raw.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["seeds"] = len(group)
        for col in numeric:
            mean, lo, hi = ci95(group[col])
            base[f"{col}_mean"] = mean
            base[f"{col}_ci95_low"] = lo
            base[f"{col}_ci95_high"] = hi
        summaries.append(base)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out, index=False)
    raw.to_csv(out.with_name(out.stem + "_seed_level.csv"), index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
