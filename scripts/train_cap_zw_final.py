from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.final import CAP_ZW_FINAL, CAPZWHashNet, FinalCAPZWConfig, PairAttackDataset, train_cap_zw_v11
from zero_watermarking.protocol import seed_everything
from zero_watermarking.synthetic import make_dataset

DEFAULT_ATTACKS = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the locked CAP-ZW research candidate.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw_final_seed42.pt")
    ap.add_argument("--history", default="experiments/results/cap_zw_final_seed42_training_history.csv")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    if args.manifest:
        frame = validate_manifest(load_manifest(args.manifest))
        frame = frame[frame["exists"]].reset_index(drop=True)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
        frame = frame.head(args.limit)
        if len(frame) < 4:
            raise SystemExit(f"Need at least 4 existing images in split={args.split}.")
        images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
    else:
        generated = make_dataset(args.images, args.size)
        images = np.stack([generated[i] for i in sorted(generated)])

    # Each image receives a unique identity label. Patient/group identifiers are
    # used for leakage-safe splitting, not as a reason to collapse two different
    # images into the same watermark identity.
    labels = np.arange(len(images), dtype=np.int64)
    dataset = PairAttackDataset(images, labels, attack_names=DEFAULT_ATTACKS, attack_views=3)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )

    config = FinalCAPZWConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        nbits=args.bits,
        device=device,
    )
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw_v11(
        model,
        loader,
        config,
        checkpoint=args.checkpoint,
        history_path=args.history,
    )
    print(
        f"Training complete: {CAP_ZW_FINAL} seed={args.seed} images={len(images)} "
        f"effective_samples={len(dataset)} bits={args.bits} epochs={len(history)} device={device}"
    )
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
