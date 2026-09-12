from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.synthetic import make_dataset
from zero_watermarking.v10 import (
    CAP_ZW_V10_VERSION,
    CAPZWHashNet,
    PairAttackDataset,
    V10Config,
    train_cap_zw_v10,
)


DEFAULT_ATTACKS = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CAP-ZW-v10 with an adaptive robustness guard.")
    parser.add_argument("--manifest", default="", help="Medical CSV manifest. Omit for synthetic smoke training.")
    parser.add_argument("--split", default="train_val")
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.36)
    parser.add_argument("--robust-target", type=float, default=0.022)
    parser.add_argument("--robust-softness", type=float, default=0.008)
    parser.add_argument("--robustness-quantile", type=float, default=0.80)
    parser.add_argument("--attack-views", type=int, default=3)
    parser.add_argument("--memory-size", type=int, default=4096)
    parser.add_argument("--memory-warmup", type=int, default=384)
    parser.add_argument("--mgda-steps", type=int, default=20)
    parser.add_argument("--topk-negatives", type=int, default=12)
    parser.add_argument("--diversity-target", type=float, default=0.36)
    parser.add_argument("--tail-target", type=float, default=0.26)
    parser.add_argument("--binary-collision-target", type=float, default=0.14)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--collision-power", type=float, default=2.0)
    parser.add_argument("--guard-target", type=float, default=0.019, help="High-quantile code-drift budget for the v10 robustness guard.")
    parser.add_argument("--guard-quantile", type=float, default=0.90)
    parser.add_argument("--guard-softness", type=float, default=0.006)
    parser.add_argument("--guard-lambda", type=float, default=1.50)
    parser.add_argument("--guard-growth", type=float, default=0.25)
    parser.add_argument("--guard-max", type=float, default=6.0)
    parser.add_argument("--guard-mean-weight", type=float, default=0.50)
    parser.add_argument("--checkpoint", default="experiments/checkpoints/cap_zw_v10.pt")
    parser.add_argument("--history", default="experiments/results/cap_zw_v10_training_history.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    import torch

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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    config = V10Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        robust_target=args.robust_target,
        robust_softness=args.robust_softness,
        robustness_quantile=args.robustness_quantile,
        attack_views=args.attack_views,
        memory_size=args.memory_size,
        memory_warmup=args.memory_warmup,
        mgda_steps=args.mgda_steps,
        topk_negatives=args.topk_negatives,
        diversity_target=args.diversity_target,
        tail_target=args.tail_target,
        binary_collision_target=args.binary_collision_target,
        uniformity_temperature=args.temperature,
        collision_power=args.collision_power,
        nbits=args.bits,
        device=device,
        robustness_guard_target=args.guard_target,
        robustness_guard_quantile=args.guard_quantile,
        robustness_guard_softness=args.guard_softness,
        robustness_guard_lambda_init=args.guard_lambda,
        robustness_guard_lambda_growth=args.guard_growth,
        robustness_guard_lambda_max=args.guard_max,
        robustness_guard_mean_weight=args.guard_mean_weight,
    )
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw_v10(
        model,
        loader,
        config,
        checkpoint=args.checkpoint,
        history_path=args.history,
    )

    print(
        f"Training complete: version={CAP_ZW_V10_VERSION} epochs={len(history)} "
        f"images={len(images)} effective_samples={len(dataset)} bits={args.bits} device={device}"
    )
    print(
        f"robust_target={args.robust_target} guard_target={args.guard_target} "
        f"guard_q={args.guard_quantile} guard_lambda={args.guard_lambda} guard_max={args.guard_max}"
    )
    print(
        f"binary_collision_target={args.binary_collision_target} tail_target={args.tail_target} "
        f"topk={args.topk_negatives} memory={args.memory_size} attack_views={args.attack_views}"
    )
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
