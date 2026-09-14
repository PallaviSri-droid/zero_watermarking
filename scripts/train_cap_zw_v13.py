from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.protocol import seed_everything
from zero_watermarking.v13 import CAPZWHashNet, PairAttackDataset, V13Config, train_cap_zw_v13

DEFAULT_ATTACKS = (
    "gaussian_noise", "salt_pepper", "gaussian_blur", "median_blur", "jpeg",
    "brightness", "contrast", "rotation", "crop_resize", "translation", "compound",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CAP-ZW-v13 Pareto-refinement research candidate.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw_v13_seed42.pt")
    ap.add_argument("--history", default="experiments/results/cap_zw_v13_seed42_training_history.csv")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
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
    dataset = PairAttackDataset(images, labels, attack_names=DEFAULT_ATTACKS, attack_views=4)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                         pin_memory=(device == "cuda"), generator=torch.Generator().manual_seed(args.seed))
    config = V13Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, nbits=args.bits, device=device)
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw_v13(model, loader, config, checkpoint=args.checkpoint, history_path=args.history)
    print(f"Training complete: CAP-ZW-v13 seed={args.seed} images={len(images)} effective_samples={len(dataset)} bits={args.bits} epochs={len(history)} device={device}")
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
