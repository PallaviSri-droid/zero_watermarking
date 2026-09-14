from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = {
    "mean_nc": "max",
    "mean_ber": "min",
    "mean_intra_hd": "min",
    "max_intra_hd": "min",
    "mean_inter_hd": "max",
    "min_inter_hd": "max",
    "auc": "max",
    "eer": "min",
    "collision_gap": "max",
    "inter_q05": "max",
    "inter_q10": "max",
    "intra_q90": "min",
    "intra_q95": "min",
    "q05_tail_gap": "max",
    "q10_tail_gap": "max",
    "exact_collision_rate_per_10k": "min",
    "ultra_near_rate_le_0.05_per_10k": "min",
    "near_collision_rate_le_0.10_per_10k": "min",
    "balance_error": "min",
    "bit_entropy": "max",
    "mean_abs_corr": "min",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise SystemExit(f"Expected one summary row in {path}, found {len(frame)}")
    frame.insert(0, "method", label)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the common-protocol CAP-ZW vs LogPolar+DINOv2+MRELBP comparison table.")
    ap.add_argument("--cap-summary", required=True)
    ap.add_argument("--hybrid-summary", required=True)
    ap.add_argument("--output", default="experiments/results/cap_vs_logpolar_dino_mrelbp/comparison.csv")
    args = ap.parse_args()

    combined = pd.concat([
        _read(Path(args.cap_summary), "CAP-ZW"),
        _read(Path(args.hybrid_summary), "LogPolar+DINOv2+MRELBP"),
    ], ignore_index=True)

    rows = []
    for metric, direction in METRICS.items():
        if metric not in combined.columns:
            continue
        values = pd.to_numeric(combined[metric], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        winner = combined.loc[values.idxmax() if direction == "max" else values.idxmin(), "method"]
        rows.append({"metric": metric, "direction": direction, "winner": winner})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    pd.DataFrame(rows).to_csv(output.with_name("metric_winners.csv"), index=False)

    display_cols = ["method"] + [m for m in METRICS if m in combined.columns]
    print(combined[display_cols].to_string(index=False))
    print(f"\nComparison written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
