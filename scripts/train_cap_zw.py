from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.synthetic import make_dataset
from zero_watermarking.training import CAPZWHashNet, PairAttackDataset, TrainConfig, train_cap_zw


DEFAULT_ATTACKS = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CAP-ZW v6 with robustness-constrained tail-collision and multi-view consistency objectives.")
    parser.add_argument("--manifest", default="", help="Medical CSV manifest. Omit for synthetic smoke training.")
    parser.add_argument("--split", default="train_val", help="Manifest split used for training.")
    parser.add_argument("--images", type=int, default=64, help="Synthetic images when --manifest is omitted.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum real training images.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--margin", type=float, default=0.38)
    parser.add_argument("--robust-target", type=float, default=0.022, help="Target normalized clean/attacked hash distance.")
    parser.add_argument("--robustness-quantile", type=float, default=0.80)
    parser.add_argument("--attack-views", type=int, default=3, help="Number of deterministic attacked views sampled per image across epochs.")
    parser.add_argument("--memory-size", type=int, default=2048)
    parser.add_argument("--memory-warmup", type=int, default=256)
    parser.add_argument("--mgda-steps", type=int, default=20)
    parser.add_argument("--topk-negatives", type=int, default=8)
    parser.add_argument("--diversity-target", type=float, default=0.40)
    parser.add_argument("--tail-target", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--collision-power", type=float, default=2.0)
    parser.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw.pt")
    parser.add_argument("--history", default="experiments/results/cap_zw_training_history.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch installation.")

    if args.manifest:
        frame = validate_manifest(load_manifest(args.manifest))
        frame = frame[frame.exists].reset_index(drop=True)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
        if len(frame) < 4:
            raise SystemExit(f"Need at least 4 existing training images in split={args.split}.")
        frame = frame.head(args.limit)
        images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
        labels = np.arange(len(images), dtype=np.int64)
    else:
        generated = make_dataset(args.images, args.size)
        images = np.stack([generated[i] for i in sorted(generated)])
        labels = np.arange(len(images), dtype=np.int64)

    dataset = PairAttackDataset(images, labels, attack_names=DEFAULT_ATTACKS, attack_views=args.attack_views)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        robust_target=args.robust_target,
        robustness_quantile=args.robustness_quantile,
        attack_views=args.attack_views,
        memory_size=args.memory_size,
        memory_warmup=args.memory_warmup,
        mgda_steps=args.mgda_steps,
        topk_negatives=args.topk_negatives,
        diversity_target=args.diversity_target,
        tail_target=args.tail_target,
        uniformity_temperature=args.temperature,
        collision_power=args.collision_power,
        nbits=args.bits,
        device=device,
    )
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw(model, loader, config, checkpoint=args.checkpoint)

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"Training complete: epochs={len(history)} images={len(images)} bits={args.bits} device={device}")
    print(f"version=CAP-ZW-v6 memory_size={args.memory_size} warmup={args.memory_warmup} topk={args.topk_negatives}")
    print(f"margin={args.margin} robust_target={args.robust_target} tail_target={args.tail_target} diversity_target={args.diversity_target}")
    print(f"temperature={args.temperature} robustness_quantile={args.robustness_quantile} attack_views={args.attack_views} mgda_steps={args.mgda_steps}")
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={history_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
