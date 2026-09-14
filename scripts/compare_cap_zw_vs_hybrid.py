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
        raise SystemExit(f"Expected exactly one summary row in {path}, found {len(frame)}")
    frame.insert(0, "method", label)
    return frame


def _value(row: pd.Series, canonical: str, aliases: tuple[str, ...] = ()):
    for key in (canonical, *aliases):
        if key in row.index and pd.notna(row[key]):
            return row[key]
    return None


def _check_contract(combined: pd.DataFrame) -> None:
    fields = {
        "split": ("split", "eval_split"),
        "images": ("images", "test_images"),
        "bits": ("bits",),
        "image_size": ("image_size",),
        "attack_grid": ("attack_grid",),
    }
    problems: list[str] = []
    for canonical, aliases in fields.items():
        normalized = []
        for _, row in combined.iterrows():
            value = _value(row, canonical, aliases[1:])
            normalized.append(str(value) if value is not None else "<missing>")
        if len(set(normalized)) > 1:
            problems.append(f"{canonical}: {normalized}")
        if normalized and normalized[0] == "<missing>":
            problems.append(f"{canonical}: missing from one or more summaries")
    if problems:
        raise SystemExit("Benchmark contract violation:\n" + "\n".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CAP-ZW and competing methods under the locked benchmark contract.")
    ap.add_argument("--input", nargs="+", metavar="LABEL=SUMMARY.csv", help="One or more labeled summary files")
    ap.add_argument("--cap-summary", help="Backward-compatible CAP-ZW summary path")
    ap.add_argument("--hybrid-summary", help="Backward-compatible LogPolar+DINOv2+MRELBP summary path")
    ap.add_argument("--output", default="experiments/results/common_comparison/comparison.csv")
    args = ap.parse_args()

    specs = list(args.input or [])
    if args.cap_summary:
        specs.append(f"CAP-ZW={args.cap_summary}")
    if args.hybrid_summary:
        specs.append(f"LogPolar+DINOv2+MRELBP={args.hybrid_summary}")
    if len(specs) < 2:
        raise SystemExit("Provide at least two summaries via --input LABEL=PATH or the legacy --cap-summary/--hybrid-summary options.")

    frames: list[pd.DataFrame] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid input {spec!r}; expected LABEL=SUMMARY.csv")
        label, path = spec.split("=", 1)
        frames.append(_read(Path(path), label))

    combined = pd.concat(frames, ignore_index=True)
    _check_contract(combined)

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
