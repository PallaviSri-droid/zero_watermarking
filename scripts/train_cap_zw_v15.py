from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.v15 import CAP_ZW_V15_VERSION, PairAttackDataset, RelationalHashNetV15, V15Config, train_cap_zw_v15
from zero_watermarking.protocol import seed_everything

DEFAULT_ATTACKS = (
    "gaussian_noise", "salt_pepper", "gaussian_blur", "median_blur", "jpeg",
    "brightness", "contrast", "rotation", "crop_resize", "translation", "compound",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CAP-ZW-v15 dual-space relational hash candidate.")
    parser.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    parser.add_argument("--split", default="train_val")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw_v15_seed42.pt")
    parser.add_argument("--history", default="experiments/results/cap_zw_v15_seed42_training_history.csv")
    args = parser.parse_args()

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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    config = V15Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, nbits=args.bits, device=device)
    model = RelationalHashNetV15(nbits=args.bits)
    history = train_cap_zw_v15(model, loader, config, checkpoint=args.checkpoint, history_path=args.history)
    print(
        f"Training complete: {CAP_ZW_V15_VERSION} seed={args.seed} images={len(images)} "
        f"effective_samples={len(dataset)} bits={args.bits} epochs={len(history)} device={device}"
    )
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
