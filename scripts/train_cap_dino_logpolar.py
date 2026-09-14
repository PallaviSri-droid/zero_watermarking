from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zero_watermarking.cap_dino_logpolar import (
    CAPDinoLogPolar,
    CAP_DINO_LP_VERSION,
    FusionConfig,
    fusion_loss,
    fusion_objective,
)
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.protocol import seed_everything
from zero_watermarking.training import PairAttackDataset

ATTACKS = (
    "gaussian_noise",
    "salt_pepper",
    "gaussian_blur",
    "median_blur",
    "jpeg",
    "brightness",
    "contrast",
    "rotation",
    "crop_resize",
    "translation",
    "compound",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CAP + DINOv2 + Log-Polar adaptive fusion.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_seed42.pt")
    ap.add_argument("--history", default="experiments/results/cap_dino_logpolar_seed42_training_history.csv")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 8:
        raise SystemExit(f"Need at least 8 existing images in split={args.split}.")

    images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
    labels = np.arange(len(images), dtype=np.int64)
    dataset = PairAttackDataset(images, labels, attack_names=ATTACKS, attack_views=4)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )

    config = FusionConfig(bits=args.bits, device=device)
    model = CAPDinoLogPolar(config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=2e-4,
    )

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        totals = {
            k: 0.0
            for k in (
                "loss",
                "robustness",
                "discrimination",
                "separation_tail",
                "entropy_loss",
                "balance",
                "decorrelation",
                "binary_collision",
                "gate_entropy",
                "gate_balance",
                "bit_entropy",
            )
        }
        gate_sum = np.zeros(3, dtype=np.float64)
        batches = 0

        for batch in loader:
            clean_x, attacked_x, labels_t = [x.to(device) for x in batch[:3]]

            # Use straight-through hard bits during training so the objective
            # sees the same binary representation that the evaluator measures.
            clean, clean_gates = model.forward_with_gate(clean_x, hard=True)
            attacked, attacked_gates = model.forward_with_gate(attacked_x, hard=True)
            gates = torch.cat([clean_gates, attacked_gates], dim=0)
            terms = fusion_objective(clean, attacked, labels_t, gates)
            loss = fusion_loss(terms)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for key in totals:
                if key in terms:
                    totals[key] += float(terms[key].detach())
            totals["loss"] += float(loss.detach())
            p = clean.detach().mean(dim=0)
            bit_entropy = -(p.clamp(1e-5, 1 - 1e-5) * torch.log2(p.clamp(1e-5, 1 - 1e-5)) +
                            (1 - p).clamp(1e-5, 1 - 1e-5) *
                            torch.log2((1 - p).clamp(1e-5, 1 - 1e-5))).mean()
            totals["bit_entropy"] += float(bit_entropy)
            gate_sum += gates.detach().mean(dim=0).cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1), **{k: v / denom for k, v in totals.items()}}
        row.update({f"gate_{i}": float(gate_sum[i] / denom) for i in range(3)})
        history.append(row)

        path = Path(args.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": CAP_DINO_LP_VERSION,
                "model": model.state_dict(),
                "config": config.__dict__,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            },
            path,
        )

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(
        f"Training complete: {CAP_DINO_LP_VERSION} seed={args.seed} images={len(images)} "
        f"effective_samples={len(dataset)} bits={args.bits} epochs={args.epochs} device={device}"
    )
    print(f"checkpoint={path.resolve()}")
    print(f"history={history_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
