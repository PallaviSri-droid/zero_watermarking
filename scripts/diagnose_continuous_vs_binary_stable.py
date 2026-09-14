from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from zero_watermarking.attacks import ATTACKS
from zero_watermarking.cap_dino_logpolar_stable import CAPDinoLogPolarStable, STABLE_VERSION, StableConfig
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest

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


def _config_from_checkpoint(payload: dict, device: str) -> StableConfig:
    raw = dict(payload.get("config", {}))
    valid = {field.name for field in fields(StableConfig)}
    cfg = StableConfig(**{k: v for k, v in raw.items() if k in valid})
    cfg.device = device
    return cfg


def pairwise_cosine_distance(x: np.ndarray) -> np.ndarray:
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = np.clip(z @ z.T, -1.0, 1.0)
    d = 1.0 - sim
    np.fill_diagonal(d, np.nan)
    return d


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sq = np.sum(z * z, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (z @ z.T), 0.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.nan)
    return d


def summarize_inter(d: np.ndarray) -> dict[str, float]:
    vals = d[np.isfinite(d)]
    return {
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "q01": float(np.quantile(vals, 0.01)),
        "q05": float(np.quantile(vals, 0.05)),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
    }


def summarize_intra(clean: np.ndarray, attacked_mean: np.ndarray, attacked_worst: np.ndarray) -> dict[str, float]:
    cosine = np.sum(
        clean / np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
        * attacked_mean / np.maximum(np.linalg.norm(attacked_mean, axis=1, keepdims=True), 1e-12),
        axis=1,
    )
    cosine_d = 1.0 - np.clip(cosine, -1.0, 1.0)
    clean_n = clean / np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
    worst_n = attacked_worst / np.maximum(np.linalg.norm(attacked_worst, axis=1, keepdims=True), 1e-12)
    worst_d = 1.0 - np.sum(clean_n * worst_n, axis=1)
    return {
        "mean": float(cosine_d.mean()),
        "q90": float(np.quantile(cosine_d, 0.90)),
        "q95": float(np.quantile(cosine_d, 0.95)),
        "max": float(cosine_d.max()),
        "worst_attack_mean": float(worst_d.mean()),
        "worst_attack_max": float(worst_d.max()),
    }


def auroc_from_distances(inter: np.ndarray, intra: np.ndarray) -> float:
    y = np.concatenate([np.zeros(intra.size, dtype=np.int8), np.ones(inter.size, dtype=np.int8)])
    s = np.concatenate([intra, inter])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose continuous fused-space collapse versus binary hash collapse.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_stable_seed42.pt")
    ap.add_argument("--out", default="experiments/results/stable_continuous_vs_binary_200")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload, device)
    model = CAPDinoLogPolarStable(cfg)
    model.load_state_dict(payload["model"])
    model.eval()

    ids: list[str] = []
    clean_fused: list[np.ndarray] = []
    clean_logits: list[np.ndarray] = []
    clean_bits: list[np.ndarray] = []
    attacked_fused: list[list[np.ndarray]] = []
    attacked_bits: list[list[np.ndarray]] = []
    details: list[dict[str, object]] = []

    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            image = load_image(row.path, args.size)
            x = torch.from_numpy(image[None, None]).to(device)
            identity = model.forward_pair(x, x)
            ids.append(str(row.image_id))
            clean_fused.append(identity["clean_fused"].cpu().numpy()[0])
            clean_logits.append(identity["clean_logits"].cpu().numpy()[0])
            clean_bits.append(identity["clean_hard"].round().to(torch.uint8).cpu().numpy()[0])

            row_att_fused: list[np.ndarray] = []
            row_att_bits: list[np.ndarray] = []
            for index, (name, kwargs) in enumerate(ATTACK_GRID):
                params = dict(kwargs)
                if name in {"gaussian_noise", "compound"}:
                    params["seed"] = int(params.get("seed", 0)) + index
                attacked = ATTACKS[name](image, **params)
                ax = torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None, None]).to(device)
                pair = model.forward_pair(x, ax)
                row_att_fused.append(pair["attacked_fused"].cpu().numpy()[0])
                row_att_bits.append(pair["attacked_hard"].round().to(torch.uint8).cpu().numpy()[0])
                details.append({
                    "image_id": str(row.image_id),
                    "attack": f"{name}_{index}",
                    "continuous_cosine_distance": float(
                        1.0 - F.cosine_similarity(pair["clean_fused"], pair["attacked_fused"], dim=-1).cpu().item()
                    ),
                    "continuous_l2_normalized": float(
                        torch.linalg.vector_norm(
                            F.normalize(pair["clean_fused"], dim=-1) - F.normalize(pair["attacked_fused"], dim=-1), dim=-1
                        ).cpu().item()
                    ),
                    "binary_hamming": float((pair["clean_hard"] >= 0.5).ne(pair["attacked_hard"] >= 0.5).float().mean().cpu().item()),
                })
            attacked_fused.append(row_att_fused)
            attacked_bits.append(row_att_bits)

    clean_fused_np = np.stack(clean_fused)
    clean_logits_np = np.stack(clean_logits)
    clean_bits_np = np.stack(clean_bits).astype(np.uint8)

    cosine_matrix = pairwise_cosine_distance(clean_fused_np)
    l2_matrix = pairwise_l2(clean_fused_np)
    logits_cosine_matrix = pairwise_cosine_distance(clean_logits_np)
    inter_cos = summarize_inter(cosine_matrix)
    inter_l2 = summarize_inter(l2_matrix)
    inter_logits = summarize_inter(logits_cosine_matrix)

    intra_fused_rows = []
    intra_logits_rows = []
    intra_binary_rows = []
    per_attack = []
    for attack_index, (name, kwargs) in enumerate(ATTACK_GRID):
        af = np.stack([attacked_fused[i][attack_index] for i in range(len(ids))])
        ab = np.stack([attacked_bits[i][attack_index] for i in range(len(ids))]).astype(np.uint8)
        clean_n = clean_fused_np / np.maximum(np.linalg.norm(clean_fused_np, axis=1, keepdims=True), 1e-12)
        af_n = af / np.maximum(np.linalg.norm(af, axis=1, keepdims=True), 1e-12)
        cd = 1.0 - np.sum(clean_n * af_n, axis=1)
        ln = clean_logits_np / np.maximum(np.linalg.norm(clean_logits_np, axis=1, keepdims=True), 1e-12)
        al = np.stack([np.asarray(v) for v in [pair for pair in []]]) if False else None
        # Logit attacks are reconstructed from the fused attack features through the same frozen hash head only
        # in the model forward above; they are not retained separately, so per-attack logit diagnostics are omitted.
        hd = (clean_bits_np != ab).mean(axis=1)
        intra_fused_rows.append(cd)
        intra_binary_rows.append(hd)
        per_attack.append({
            "attack": f"{name}_{attack_index}",
            "continuous_mean": float(cd.mean()),
            "continuous_q90": float(np.quantile(cd, 0.90)),
            "continuous_q95": float(np.quantile(cd, 0.95)),
            "continuous_max": float(cd.max()),
            "binary_mean": float(hd.mean()),
            "binary_q90": float(np.quantile(hd, 0.90)),
            "binary_q95": float(np.quantile(hd, 0.95)),
            "binary_max": float(hd.max()),
        })

    intra_fused = np.concatenate(intra_fused_rows)
    intra_binary = np.concatenate(intra_binary_rows)
    continuous_auc = auroc_from_distances(cosine_matrix[np.isfinite(cosine_matrix)], intra_fused)

    summary = {
        "version": str(payload.get("version", STABLE_VERSION)),
        "images": len(ids),
        "bits": int(clean_bits_np.shape[1]),
        "fused_dim": int(clean_fused_np.shape[1]),
        "logit_dim": int(clean_logits_np.shape[1]),
        "continuous_inter_cos_mean": inter_cos["mean"],
        "continuous_inter_cos_min": inter_cos["min"],
        "continuous_inter_cos_q01": inter_cos["q01"],
        "continuous_inter_cos_q05": inter_cos["q05"],
        "continuous_inter_cos_q10": inter_cos["q10"],
        "continuous_inter_cos_q50": inter_cos["q50"],
        "continuous_inter_l2_mean": inter_l2["mean"],
        "continuous_inter_l2_q05": inter_l2["q05"],
        "continuous_inter_l2_q10": inter_l2["q10"],
        "logit_inter_cos_mean": inter_logits["mean"],
        "logit_inter_cos_q05": inter_logits["q05"],
        "logit_inter_cos_q10": inter_logits["q10"],
        "continuous_intra_cos_mean": float(intra_fused.mean()),
        "continuous_intra_cos_q90": float(np.quantile(intra_fused, 0.90)),
        "continuous_intra_cos_q95": float(np.quantile(intra_fused, 0.95)),
        "continuous_intra_cos_max": float(intra_fused.max()),
        "binary_intra_hamming_mean": float(intra_binary.mean()),
        "binary_intra_hamming_q90": float(np.quantile(intra_binary, 0.90)),
        "binary_intra_hamming_q95": float(np.quantile(intra_binary, 0.95)),
        "binary_intra_hamming_max": float(intra_binary.max()),
        "continuous_vs_intra_auc": continuous_auc,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_attack).to_csv(out / "per_attack.csv", index=False)
    pd.DataFrame(details).to_csv(out / "attack_details.csv", index=False)

    print(pd.Series(summary).to_string())
    print("\nInterpretation aid:")
    print("  High continuous inter distance + high binary collisions -> hashing/quantization bottleneck.")
    print("  Low continuous inter distance + binary collisions -> representation bottleneck.")
    print("  Compare continuous inter Q05/Q10 against continuous intra Q95, not only means.")
    print(f"Results written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
