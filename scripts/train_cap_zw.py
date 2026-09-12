from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.synthetic import make_dataset
from zero_watermarking.training import CAPZWHashNet, PairAttackDataset, TrainConfig, train_cap_zw


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CAP-ZW on synthetic or real manifest data.")
    parser.add_argument("--manifest", default="", help="Optional populated medical CSV manifest.")
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw.pt")
    args = parser.parse_args()

    if args.manifest:
        frame = validate_manifest(load_manifest(args.manifest))
        frame = frame[frame.exists].reset_index(drop=True)
        if len(frame) < 4:
            raise SystemExit("Populate the manifest with at least 4 existing images before real-data training.")
        images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
        labels = np.arange(len(images), dtype=np.int64)
    else:
        generated = make_dataset(args.images, args.size)
        images = np.stack([generated[i] for i in sorted(generated)])
        labels = np.arange(len(images), dtype=np.int64)

    dataset = PairAttackDataset(images, labels, attack_name="gaussian_noise")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, nbits=args.bits)
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw(model, loader, config, checkpoint=args.checkpoint)
    print(f"Training complete: epochs={len(history)} checkpoint={Path(args.checkpoint)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
