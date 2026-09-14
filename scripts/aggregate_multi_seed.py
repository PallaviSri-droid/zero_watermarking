from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zero_watermarking.research_contract import assert_compatible_summaries


def ci95(values: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if x.size == 1:
        return mean, float("nan"), float("nan")
    from scipy.stats import t
    half = float(t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))
    return mean, mean - half, mean + half


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seed-level benchmark summaries with protocol checks.")
    parser.add_argument("inputs", nargs="+", help="Seed summary.csv files or directories containing summary.csv")
    parser.add_argument("--out", default="experiments/results/multi_seed_summary.csv")
    parser.add_argument("--require-seeds", type=int, default=1, help="Fail unless at least this many seed summaries are supplied")
    args = parser.parse_args()

    rows = []
    reference: dict | None = None
    for raw_path in args.inputs:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "summary.csv"
        if not path.exists():
            raise SystemExit(f"Missing summary: {path}")
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise SystemExit(f"Expected exactly one summary row in {path}, found {len(frame)}")
        row = frame.iloc[0].to_dict()
        if reference is None:
            reference = row
        else:
            assert_compatible_summaries(reference, row)
        row["source"] = str(path)
        rows.append(row)

    if len(rows) < args.require_seeds:
        raise SystemExit(f"Expected at least {args.require_seeds} seed summaries, found {len(rows)}")

    raw = pd.DataFrame(rows)
    numeric = [c for c in raw.columns if c not in {"version", "split", "source"} and pd.api.types.is_numeric_dtype(raw[c])]
    group_cols = [c for c in ("method", "version", "split", "bits", "manifest_id", "attack_grid_id") if c in raw.columns]
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
            base[f"{col}_std"] = float(pd.to_numeric(group[col], errors="coerce").std(ddof=1)) if len(group) > 1 else float("nan")
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
