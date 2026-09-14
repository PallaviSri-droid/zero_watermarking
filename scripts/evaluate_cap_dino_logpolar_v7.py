from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.cap_dino_logpolar_v7 import CAPDinoLogPolarV7, V7Config, CAP_DINO_LP_V7_VERSION
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.metrics import bit_balance, bit_entropy, mean_abs_corr, evaluate_hash_bank

ATTACK_GRID = (
    ("gaussian_noise", {"sigma": 0.03, "seed": 11}),
    ("gaussian_noise", {"sigma": 0.08, "seed": 17}),
    ("gaussian_blur", {"sigma": 1.0}),
    ("gaussian_blur", {"sigma": 2.0}),
    ("jpeg", {"quality": 70}),
    ("jpeg", {"quality": 40}),
    ("rotation", {"degrees": 5.0}),
    ("rotation", {"degrees": 10.0}),
    ("crop_resize", {"fraction": 0.05}),
    ("translation", {"pixels": 3}),
    ("compound", {"seed": 23}),
)


def load_model(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = V7Config(**payload.get("config", {}))
    cfg.device = device
    model = CAPDinoLogPolarV7(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate CAP-DINO-LogPolar v7 on the locked benchmark.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_v7_seed42.pt")
    ap.add_argument("--out", default="experiments/results/cap_dino_logpolar_v7_eval_200")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--near-threshold", type=float, default=0.10)
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 2:
        raise SystemExit("Need at least 2 images.")

    model, metadata = load_model(checkpoint, device)
    clean_bank: dict[str, np.ndarray] = {}
    attacked_bank: dict[str, dict[str, np.ndarray]] = {}
    gate_bank: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            image = load_image(row.path, args.size)
            x = torch.from_numpy(image[None, None]).to(device)
            clean, gates = model.forward_with_gate(x, hard=True) if hasattr(model, "forward_with_gate") else (model.forward(x, hard=True), model.gate(*model.encode_branches(x)))
            clean_bank[row.image_id] = clean.round().to(torch.uint8).cpu().numpy()[0]
            gate_bank[row.image_id] = gates.cpu().numpy()[0]
            attacked_bank[row.image_id] = {}
            for index, (attack_name, kwargs) in enumerate(ATTACK_GRID):
                params = dict(kwargs)
                if attack_name in {"gaussian_noise", "compound"}:
                    params["seed"] = int(params.get("seed", 0)) + index
                attacked = ATTACKS[attack_name](image, **params)
                ax = torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None, None]).to(device)
                code = model.forward(ax, hard=True)
                attacked_bank[row.image_id][f"{attack_name}_{index}"] = code.round().to(torch.uint8).cpu().numpy()[0]

    result = evaluate_hash_bank(clean_bank, attacked_bank)
    matrix = np.stack([clean_bank[k] for k in sorted(clean_bank)])
    entropy, _ = bit_entropy(matrix)
    stats = result["collision_statistics"]
    gates = np.stack([gate_bank[k] for k in sorted(gate_bank)])
    gate_entropy_values = -(np.clip(gates, 1e-8, 1.0) * np.log(np.clip(gates, 1e-8, 1.0))).sum(axis=1)
    dominant = np.argmax(gates, axis=1)
    summary = {
        "version": str(metadata.get("version", CAP_DINO_LP_V7_VERSION)),
        "method": "CAP-DINO-LogPolar-v7-AntiDominanceMixer",
        "split": args.split,
        "images": len(clean_bank),
        "bits": int(matrix.shape[1]),
        "image_size": args.size,
        "mean_nc": result["mean_nc"],
        "mean_ber": result["mean_ber"],
        "mean_intra_hd": result["mean_intra_hd"],
        "max_intra_hd": result["max_intra_hd"],
        "mean_inter_hd": result["mean_inter_hd"],
        "min_inter_hd": result["min_inter_hd"],
        "collision_gap": result["collision_gap"],
        "auc": result["auc"],
        "eer": result["eer"],
        "balance_error": bit_balance(matrix),
        "bit_entropy": entropy,
        "mean_abs_corr": mean_abs_corr(matrix),
        "negative_pairs": stats["negative_pairs"],
        "exact_collision_pairs": stats["exact_collision_pairs"],
        "exact_collision_rate_per_10k": stats["exact_collision_rate_per_10k"],
        "inter_q01": stats["inter_q01"],
        "inter_q05": stats["inter_q05"],
        "inter_q10": stats["inter_q10"],
        "intra_q90": stats["intra_q90"],
        "intra_q95": stats["intra_q95"],
        "q05_tail_gap": stats["q05_tail_gap"],
        "q10_tail_gap": stats["q10_tail_gap"],
        "ultra_near_pairs_le_0.05": stats["collision_pairs_le_0.05"],
        "ultra_near_rate_le_0.05_per_10k": stats["collision_rate_le_0.05_per_10k"],
        "near_collision_pairs_le_0.10": stats["collision_pairs_le_0.10"],
        "near_collision_rate_le_0.10_per_10k": stats["collision_rate_le_0.10_per_10k"],
        "gate_cap_mean": float(gates[:, 0].mean()),
        "gate_dino_mean": float(gates[:, 1].mean()),
        "gate_logpolar_mean": float(gates[:, 2].mean()),
        "gate_cap_std": float(gates[:, 0].std()),
        "gate_dino_std": float(gates[:, 1].std()),
        "gate_logpolar_std": float(gates[:, 2].std()),
        "gate_entropy_mean": float(gate_entropy_values.mean()),
        "gate_entropy_std": float(gate_entropy_values.std()),
        "fraction_cap_dominant": float(np.mean(dominant == 0)),
        "fraction_dino_dominant": float(np.mean(dominant == 1)),
        "fraction_logpolar_dominant": float(np.mean(dominant == 2)),
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(result["details"], columns=["image_id", "attack", "hamming", "nc"]).to_csv(out / "attack_details.csv", index=False)
    gate_rows = [{"image_id": k, "gate_cap": float(v[0]), "gate_dino": float(v[1]), "gate_logpolar": float(v[2])} for k, v in sorted(gate_bank.items())]
    pd.DataFrame(gate_rows).to_csv(out / "gates.csv", index=False)
    ids = sorted(clean_bank)
    pairs: list[dict[str, object]] = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            d = float(np.mean(clean_bank[left] != clean_bank[right]))
            if d <= args.near_threshold:
                pairs.append({"image_a": left, "image_b": right, "hamming": d, "exact_collision": d == 0.0, "near_collision": True})
    pd.DataFrame(pairs, columns=["image_a", "image_b", "hamming", "exact_collision", "near_collision"]).to_csv(out / "collision_pairs.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Results written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
