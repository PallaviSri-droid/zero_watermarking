from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.cap_dino_logpolar import FusionConfig
from zero_watermarking.cap_dino_logpolar_v5 import CAPDinoLogPolarV5, V5Config, CAP_DINO_LP_V5_VERSION
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


def load_model(checkpoint: Path, device: str) -> tuple[CAPDinoLogPolarV5, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = V5Config(**payload.get("config", {}))
    cfg.device = device
    model = CAPDinoLogPolarV5(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate CAP-DINO-LogPolar v5 on the locked benchmark.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_v5_seed42.pt")
    ap.add_argument("--out", default="experiments/results/cap_dino_logpolar_v5_eval")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--near-threshold", type=float, default=0.10)
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 2:
        raise SystemExit("Need at least 2 images.")

    model, metadata = load_model(Path(args.checkpoint), device)
    clean_bank: dict[str, np.ndarray] = {}
    attacked_bank: dict[str, dict[str, np.ndarray]] = {}
    gate_bank: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            image = load_image(row.path, args.size)
            x = torch.from_numpy(image[None, None]).to(device)
            cap, dino, lp = model.encode_branches(x)
            gates = model.gate(cap, dino, lp)
            fused = model.fuse_from_branches(cap, dino, lp, gates)
            code = model.quantizer(model.fusion(fused), hard=True)
            clean_bank[row.image_id] = code.round().to(torch.uint8).cpu().numpy()[0]
            gate_bank[row.image_id] = gates.cpu().numpy()[0]
            attacked_bank[row.image_id] = {}
            # Evaluate attacks with the model's normal clean-image routing;
            # the training-time shared-gate path is mirrored here by reusing the
            # clean gate for every attacked view of the same image.
            for index, (attack_name, kwargs) in enumerate(ATTACK_GRID):
                params = dict(kwargs)
                if attack_name in {"gaussian_noise", "compound"}:
                    params["seed"] = int(params.get("seed", 0)) + index
                attacked = ATTACKS[attack_name](image, **params)
                ax = torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None, None]).to(device)
                acap, adino, alp = model.encode_branches(ax)
                afused = model.fuse_from_branches(acap, adino, alp, gates)
                acode = model.quantizer(model.fusion(afused), hard=True)
                attacked_bank[row.image_id][f"{attack_name}_{index}"] = acode.round().to(torch.uint8).cpu().numpy()[0]

    result = evaluate_hash_bank(clean_bank, attacked_bank)
    matrix = np.stack([clean_bank[k] for k in sorted(clean_bank)])
    entropy, _ = bit_entropy(matrix)
    stats = result["collision_statistics"]
    gate_matrix = np.stack([gate_bank[k] for k in sorted(gate_bank)])
    gate_ent = -(np.clip(gate_matrix, 1e-8, 1) * np.log(np.clip(gate_matrix, 1e-8, 1))).sum(axis=1)
    winner = np.argmax(gate_matrix, axis=1)

    summary = {
        "version": str(metadata.get("version", CAP_DINO_LP_V5_VERSION)),
        "method": "CAP-DINO-LogPolar-v5-SharedGate",
        "split": args.split, "images": len(clean_bank), "bits": int(matrix.shape[1]), "image_size": args.size,
        "mean_nc": result["mean_nc"], "mean_ber": result["mean_ber"],
        "mean_intra_hd": result["mean_intra_hd"], "max_intra_hd": result["max_intra_hd"],
        "mean_inter_hd": result["mean_inter_hd"], "min_inter_hd": result["min_inter_hd"],
        "collision_gap": result["collision_gap"], "auc": result["auc"], "eer": result["eer"],
        "balance_error": bit_balance(matrix), "bit_entropy": entropy, "mean_abs_corr": mean_abs_corr(matrix),
        "negative_pairs": stats["negative_pairs"], "exact_collision_pairs": stats["exact_collision_pairs"],
        "exact_collision_rate_per_10k": stats["exact_collision_rate_per_10k"],
        "inter_q01": stats["inter_q01"], "inter_q05": stats["inter_q05"], "inter_q10": stats["inter_q10"],
        "intra_q90": stats["intra_q90"], "intra_q95": stats["intra_q95"],
        "q05_tail_gap": stats["q05_tail_gap"], "q10_tail_gap": stats["q10_tail_gap"],
        "ultra_near_pairs_le_0.05": stats["collision_pairs_le_0.05"],
        "ultra_near_rate_le_0.05_per_10k": stats["collision_rate_le_0.05_per_10k"],
        "near_collision_pairs_le_0.10": stats["collision_pairs_le_0.10"],
        "near_collision_rate_le_0.10_per_10k": stats["collision_rate_le_0.10_per_10k"],
        "gate_cap_mean": float(gate_matrix[:, 0].mean()), "gate_dino_mean": float(gate_matrix[:, 1].mean()),
        "gate_logpolar_mean": float(gate_matrix[:, 2].mean()), "gate_entropy_mean": float(gate_ent.mean()),
        "gate_entropy_std": float(gate_ent.std()), "gate_cap_std": float(gate_matrix[:, 0].std()),
        "gate_dino_std": float(gate_matrix[:, 1].std()), "gate_logpolar_std": float(gate_matrix[:, 2].std()),
        "fraction_cap_dominant": float(np.mean(winner == 0)), "fraction_dino_dominant": float(np.mean(winner == 1)),
        "fraction_logpolar_dominant": float(np.mean(winner == 2)),
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
            distance = float(np.mean(clean_bank[left] != clean_bank[right]))
            if distance <= args.near_threshold:
                pairs.append({"image_a": left, "image_b": right, "hamming": distance,
                              "exact_collision": distance == 0.0, "near_collision": True})
    pd.DataFrame(pairs, columns=["image_a", "image_b", "hamming", "exact_collision", "near_collision"]).to_csv(out / "collision_pairs.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Results written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
