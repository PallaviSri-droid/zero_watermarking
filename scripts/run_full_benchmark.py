from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.baselines import method_registry
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.deep_baselines import build_deep_registry
from zero_watermarking.metrics import bit_balance, bit_entropy, hamming, mean_abs_corr, nc, roc_stats
from zero_watermarking.protocol import DEFAULT_ATTACK_GRID, seed_everything


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the common medical zero-watermarking benchmark.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--deep", action="store_true", help="Also run optional pretrained deep baselines.")
    args = ap.parse_args()

    if args.limit < 2:
        raise SystemExit("--limit must be at least 2 for discriminability metrics.")

    seed_everything(42)
    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].head(args.limit).reset_index(drop=True)
    if len(frame) < 2:
        raise SystemExit("The manifest must contain at least 2 existing images before benchmarking.")

    images = {row.image_id: load_image(row.path, args.size) for row in frame.itertuples(index=False)}
    registry = method_registry(args.bits)
    if args.deep:
        registry.update({k: v for k, v in build_deep_registry(args.bits).items() if v is not None})

    attacks = [(spec.name, spec.parameter, value) for spec in DEFAULT_ATTACK_GRID for value in spec.values]
    rows: list[dict] = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in registry.items():
        clean = {image_id: np.asarray(fn(image), dtype=np.uint8) for image_id, image in images.items()}
        bank: list[dict] = []
        for image_id, image in images.items():
            for attack, parameter, value in attacks:
                attacked = ATTACKS[attack](image, **{parameter: value})
                attacked_hash = np.asarray(fn(attacked), dtype=np.uint8)
                bank.append({
                    "method": name,
                    "image_id": image_id,
                    "attack": attack,
                    "strength": value,
                    "hamming": hamming(clean[image_id], attacked_hash),
                    "nc": nc(clean[image_id], attacked_hash),
                })

        detail = pd.DataFrame(bank)
        detail.to_csv(out / f"{name.replace('/', '_')}_detail.csv", index=False)
        ids = sorted(clean)
        bits = np.stack([clean[i] for i in ids])
        genuine = 1.0 - detail["hamming"].to_numpy(dtype=float)
        impostor = np.array(
            [1.0 - hamming(clean[ids[i]], clean[ids[j]]) for i in range(len(ids)) for j in range(i + 1, len(ids))],
            dtype=float,
        )
        roc = roc_stats(genuine, impostor)
        entropy, _ = bit_entropy(bits)
        max_intra = float(detail["hamming"].max())
        min_inter = float(np.min(1.0 - impostor))
        rows.append({
            "method": name,
            "mean_nc": float(detail["nc"].mean()),
            "mean_ber": float(detail["hamming"].mean()),
            "max_intra_hd": max_intra,
            "min_inter_hd": min_inter,
            "collision_gap": min_inter - max_intra,
            "auc": float(roc["auc"]),
            "eer": float(roc["eer"]),
            "balance_error": float(bit_balance(bits)),
            "bit_entropy": float(entropy),
            "mean_abs_corr": float(mean_abs_corr(bits)),
        })

    summary = pd.DataFrame(rows).sort_values(["auc", "mean_nc"], ascending=False)
    summary.to_csv(out / "journal_benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
