from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zero_watermarking.research_contract import assert_compatible_summaries


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
    "inter_q01": "max",
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
        raise SystemExit(f"Expected exactly one summary row in {path}, found {len(frame)}")
    frame.insert(0, "method", label)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare methods under the locked publication benchmark contract.")
    ap.add_argument(
        "--input", nargs="+", action="append", metavar="LABEL=SUMMARY.csv",
        required=True, help="One or more labeled summary files; repeat --input for additional groups.",
    )
    ap.add_argument("--output", default="experiments/results/common_comparison/comparison.csv")
    args = ap.parse_args()

    specs = [item for group in args.input for item in group]
    if len(specs) < 2:
        raise SystemExit("Provide at least two inputs via --input LABEL=SUMMARY.csv")

    frames: list[pd.DataFrame] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid input {spec!r}; expected LABEL=SUMMARY.csv")
        label, path = spec.split("=", 1)
        frames.append(_read(Path(path), label))

    reference = frames[0].iloc[0].to_dict()
    for frame in frames[1:]:
        assert_compatible_summaries(reference, frame.iloc[0].to_dict())

    combined = pd.concat(frames, ignore_index=True)
    winner_rows = []
    for metric, direction in METRICS.items():
        if metric not in combined.columns:
            continue
        values = pd.to_numeric(combined[metric], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        index = valid.idxmax() if direction == "max" else valid.idxmin()
        winner_rows.append({
            "metric": metric,
            "direction": direction,
            "winner": str(combined.loc[index, "method"]),
            "winning_value": float(combined.loc[index, metric]),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(output.with_name("metric_winners.csv"), index=False)

    display_cols = ["method"] + [m for m in METRICS if m in combined.columns]
    print(combined[display_cols].to_string(index=False))
    print("\nMetric winners:")
    print(winners.to_string(index=False) if not winners.empty else "No comparable metrics found.")
    print(f"\nComparison written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
