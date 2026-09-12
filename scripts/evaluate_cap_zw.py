from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.learned import HashEncoder
from zero_watermarking.metrics import bit_balance, bit_entropy, mean_abs_corr, evaluate_hash_bank
from zero_watermarking.training import CAPZWHashNet


DEFAULT_ATTACK_GRID = (
    ("gaussian_noise", {"sigma": 0.03, "seed": 11}),
    ("gaussian_noise", {"sigma": 0.08, "seed": 17}),
    ("blur", {"sigma": 1.0}),
    ("blur", {"sigma": 2.0}),
    ("jpeg", {"quality": 70}),
    ("jpeg", {"quality": 40}),
    ("rotation", {"angle": 5.0}),
    ("rotation", {"angle": 10.0}),
    ("crop_resize", {"fraction": 0.90}),
    ("translation", {"dx": 3, "dy": 2}),
    ("compound", {"seed": 23}),
)


def load_model(checkpoint: Path, bits: int, device: str) -> CAPZWHashNet:
    model = CAPZWHashNet(nbits=bits)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained CAP-ZW model on a held-out medical split.")
    parser.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw.pt")
    parser.add_argument("--out", default="experiments/results/cap_zw_test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame.exists].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 2:
        raise SystemExit(f"Need at least 2 images in split={args.split} for discrimination evaluation.")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    model = load_model(checkpoint, args.bits, device)

    clean_bank: dict[str, np.ndarray] = {}
    attacked_bank: dict[str, dict[str, np.ndarray]] = {}
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            image = load_image(row.path, args.size)
            x = torch.from_numpy(image[None, None]).to(device)
            clean_hash = model(x, hard=True).round().to(torch.uint8).cpu().numpy()[0]
            clean_bank[row.image_id] = clean_hash
            attacked_bank[row.image_id] = {}
            for index, (attack_name, kwargs) in enumerate(DEFAULT_ATTACK_GRID):
                params = dict(kwargs)
                if attack_name == "gaussian_noise":
                    params["seed"] = int(params.get("seed", 0)) + index
                if attack_name == "compound":
                    params["seed"] = int(params.get("seed", 0)) + index
                attacked = ATTACKS[attack_name](image, **params)
                ax = torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None, None]).to(device)
                attacked_hash = model(ax, hard=True).round().to(torch.uint8).cpu().numpy()[0]
                attacked_bank[row.image_id][f"{attack_name}_{index}"] = attacked_hash

    result = evaluate_hash_bank(clean_bank, attacked_bank)
    bits_matrix = np.stack([clean_bank[key] for key in sorted(clean_bank)])
    entropy, _ = bit_entropy(bits_matrix)
    summary = {
        "split": args.split,
        "images": len(clean_bank),
        "bits": args.bits,
        "mean_nc": result["mean_nc"],
        "mean_ber": result["mean_ber"],
        "mean_intra_hd": result["mean_intra_hd"],
        "max_intra_hd": result["max_intra_hd"],
        "mean_inter_hd": result["mean_inter_hd"],
        "min_inter_hd": result["min_inter_hd"],
        "collision_gap": result["collision_gap"],
        "auc": result["auc"],
        "eer": result["eer"],
        "balance_error": bit_balance(bits_matrix),
        "bit_entropy": entropy,
        "mean_abs_corr": mean_abs_corr(bits_matrix),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(result["details"], columns=["image_id", "attack", "hamming", "nc"]).to_csv(out / "attack_details.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Results written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
