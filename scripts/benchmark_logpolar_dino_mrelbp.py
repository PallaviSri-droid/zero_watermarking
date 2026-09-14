from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.logpolar_dino_mrelbp import HybridConfig, LogPolarDinoMRELBP
from zero_watermarking.metrics import evaluate_hash_bank
from zero_watermarking.protocol import DEFAULT_ATTACK_GRID, attack_grid, seed_everything
from zero_watermarking.research_contract import attack_grid_fingerprint, manifest_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark LogPolar + DINOv2 + MRELBP under the locked CAP-ZW protocol.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="train_val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--fit-limit", type=int, default=1000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--output", default="experiments/results/logpolar_dino_mrelbp")
    args = ap.parse_args()

    seed_everything(args.seed)
    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].copy()
    if "split" not in frame.columns:
        raise SystemExit("Manifest must contain a split column for a publication comparison.")
    fit_frame = frame[frame["split"].astype(str).eq(args.fit_split)].head(args.fit_limit)
    test_frame = frame[frame["split"].astype(str).eq(args.split)].head(args.limit)
    if len(fit_frame) < 8 or len(test_frame) < 2:
        raise SystemExit("Insufficient fitted/reference or test images.")

    fit_images = np.stack([load_image(row.path, args.size) for row in fit_frame.itertuples(index=False)])
    test_images = np.stack([load_image(row.path, args.size) for row in test_frame.itertuples(index=False)])

    config = HybridConfig(bits=args.bits, device=args.device, seed=args.seed)
    t0 = time.perf_counter()
    method = LogPolarDinoMRELBP(config)
    method.fit(fit_images)
    fit_seconds = time.perf_counter() - t0

    clean_hashes = method.transform(test_images)
    attacked_bank: dict[str, dict[str, np.ndarray]] = {}
    clean_bank: dict[str, np.ndarray] = {}
    t1 = time.perf_counter()
    for idx, row in enumerate(test_frame.itertuples(index=False)):
        image_id = str(row.image_id)
        clean_bank[image_id] = clean_hashes[idx]
        attacked_bank[image_id] = {}
        image = test_images[idx]
        generated = attack_grid(image, DEFAULT_ATTACK_GRID, seed=args.seed + idx)
        for attack_name, by_strength in generated.items():
            for strength, attacked in by_strength.items():
                attacked_bank[image_id][f"{attack_name}_{strength:g}"] = method.transform(attacked[None, ...])[0]
    eval_seconds = time.perf_counter() - t1

    metrics = evaluate_hash_bank(clean_bank, attacked_bank)
    collision = metrics.pop("collision_statistics")
    out = {
        "method": "LogPolar+DINOv2+MRELBP",
        "seed": args.seed,
        "fit_split": args.fit_split,
        "split": args.split,
        "images": len(test_images),
        "fit_images": len(fit_images),
        "bits": args.bits,
        "image_size": args.size,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": eval_seconds,
        "manifest_id": manifest_fingerprint(args.manifest),
        "attack_grid_id": attack_grid_fingerprint(DEFAULT_ATTACK_GRID),
        **{k: float(v) for k, v in metrics.items() if isinstance(v, (float, int, np.floating, np.integer))},
        **collision,
        "attack_grid": json.dumps([spec.__dict__ for spec in DEFAULT_ATTACK_GRID], sort_keys=True, default=str),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(output / f"summary_seed_{args.seed}.csv", index=False)
    pd.DataFrame(metrics["details"], columns=["image_id", "attack_name", "hd", "nc"]).to_csv(
        output / f"details_seed_{args.seed}.csv", index=False
    )
    (output / f"config_seed_{args.seed}.json").write_text(
        json.dumps(
            {
                "method": config.__dict__,
                "attack_grid": [spec.__dict__ for spec in DEFAULT_ATTACK_GRID],
                "manifest": args.manifest,
                "manifest_id": manifest_fingerprint(args.manifest),
                "attack_grid_id": attack_grid_fingerprint(DEFAULT_ATTACK_GRID),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in out.items() if k != "attack_grid"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
