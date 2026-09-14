from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.baselines import method_registry
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.deep_baselines import build_deep_registry
from zero_watermarking.metrics import bit_balance, bit_entropy, evaluate_hash_bank, mean_abs_corr
from zero_watermarking.protocol import DEFAULT_ATTACK_GRID, attack_grid, seed_everything


def _numeric_summary(result: dict) -> dict[str, float | int]:
    collision = result["collision_statistics"]
    return {
        "mean_nc": float(result["mean_nc"]),
        "mean_ber": float(result["mean_ber"]),
        "mean_intra_hd": float(result["mean_intra_hd"]),
        "max_intra_hd": float(result["max_intra_hd"]),
        "mean_inter_hd": float(result["mean_inter_hd"]),
        "min_inter_hd": float(result["min_inter_hd"]),
        "collision_gap": float(result["collision_gap"]),
        "auc": float(result["auc"]),
        "eer": float(result["eer"]),
        "inter_q01": float(collision["inter_q01"]),
        "inter_q05": float(collision["inter_q05"]),
        "inter_q10": float(collision["inter_q10"]),
        "intra_q90": float(collision["intra_q90"]),
        "intra_q95": float(collision["intra_q95"]),
        "q05_tail_gap": float(collision["q05_tail_gap"]),
        "q10_tail_gap": float(collision["q10_tail_gap"]),
        "negative_pairs": int(collision["negative_pairs"]),
        "exact_collision_pairs": int(collision["exact_collision_pairs"]),
        "exact_collision_rate_per_10k": float(collision["exact_collision_rate_per_10k"]),
        "ultra_near_pairs_le_0.05": int(collision["collision_pairs_le_0.05"]),
        "ultra_near_rate_le_0.05_per_10k": float(collision["collision_rate_le_0.05_per_10k"]),
        "near_collision_pairs_le_0.10": int(collision["collision_pairs_le_0.10"]),
        "near_collision_rate_le_0.10_per_10k": float(collision["collision_rate_le_0.10_per_10k"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the locked common medical zero-watermarking benchmark.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="experiments/results/common_benchmark")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deep", action="store_true", help="Also run optional pretrained deep baselines.")
    args = ap.parse_args()

    if args.limit < 2:
        raise SystemExit("--limit must be at least 2 for discriminability metrics.")

    seed_everything(args.seed)
    frame = validate_manifest(load_manifest(args.manifest))
    if "split" not in frame.columns:
        raise SystemExit("Publication benchmark requires a split column in the manifest.")
    frame = frame[frame["exists"]].copy()
    frame = frame[frame["split"].astype(str).eq(args.split)].head(args.limit).reset_index(drop=True)
    if len(frame) < 2:
        raise SystemExit(f"Need at least 2 existing images in split={args.split}.")

    images = {str(row.image_id): load_image(row.path, args.size) for row in frame.itertuples(index=False)}
    registry = method_registry(args.bits)
    if args.deep:
        registry.update({k: v for k, v in build_deep_registry(args.bits).items() if v is not None})

    specs_json = json.dumps([spec.__dict__ for spec in DEFAULT_ATTACK_GRID], sort_keys=True)
    rows: list[dict[str, object]] = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in registry.items():
        clean_bank = {image_id: np.asarray(fn(image), dtype=np.uint8) for image_id, image in images.items()}
        attacked_bank: dict[str, dict[str, np.ndarray]] = {}
        for image_id, image in images.items():
            attacked_bank[image_id] = {}
            generated = attack_grid(image, DEFAULT_ATTACK_GRID, seed=args.seed)
            for attack_name, by_strength in generated.items():
                for strength, attacked in by_strength.items():
                    key = f"{attack_name}_{strength:g}"
                    attacked_bank[image_id][key] = np.asarray(fn(attacked), dtype=np.uint8)

        result = evaluate_hash_bank(clean_bank, attacked_bank)
        bits_matrix = np.stack([clean_bank[key] for key in sorted(clean_bank)])
        summary = {
            "method": name,
            "seed": args.seed,
            "split": args.split,
            "images": len(clean_bank),
            "bits": args.bits,
            "image_size": args.size,
            "attack_grid": specs_json,
            "balance_error": float(bit_balance(bits_matrix)),
            "bit_entropy": float(bit_entropy(bits_matrix)[0]),
            "mean_abs_corr": float(mean_abs_corr(bits_matrix)),
            **_numeric_summary(result),
        }
        rows.append(summary)

        pd.DataFrame(result["details"], columns=["image_id", "attack", "hamming", "nc"]).to_csv(
            out / f"{name.replace('/', '_')}_details.csv", index=False
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "journal_benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nResults written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
