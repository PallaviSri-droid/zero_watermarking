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


def _bool_pair(ap: argparse.ArgumentParser, name: str, default: bool) -> None:
    ap.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", default=default)
    ap.add_argument("--no-" + name.lstrip("-").replace("-", "_"), dest=name.lstrip("-").replace("-", "_"), action="store_false")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the locked CAP-ZW research candidate or a pre-registered ablation.")
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
    ap.add_argument("--experiment-name", default=CAP_ZW_FINAL)
    ap.add_argument("--lambda-robust", type=float, default=None)
    ap.add_argument("--lambda-tail", type=float, default=None)
    ap.add_argument("--lambda-diversity", type=float, default=None)
    ap.add_argument("--lambda-balance", type=float, default=None)
    ap.add_argument("--lambda-corr", type=float, default=None)
    ap.add_argument("--lambda-entropy", type=float, default=None)
    ap.add_argument("--lambda-uniformity", type=float, default=None)
    ap.add_argument("--lambda-consistency", type=float, default=None)
    ap.add_argument("--lambda-binary-collision", type=float, default=None)
    ap.add_argument("--topk-negatives", type=int, default=None)
    ap.add_argument("--binary-collision-target", type=float, default=None)
    ap.add_argument("--guard-batch-fraction", type=float, default=None)
    ap.add_argument("--guard-lambda", type=float, default=None)
    ap.add_argument("--guard-growth", type=float, default=None)
    ap.add_argument("--guard-max", type=float, default=None)
    ap.add_argument("--no-selective-guard", dest="enable_selective_guard", action="store_false")
    ap.set_defaults(enable_selective_guard=True)
    ap.add_argument("--no-hard-negative-mining", dest="enable_hard_negative_mining", action="store_false")
    ap.set_defaults(enable_hard_negative_mining=True)
    ap.add_argument("--no-memory-bank", dest="enable_memory_bank", action="store_false")
    ap.set_defaults(enable_memory_bank=True)
    ap.add_argument("--no-mgda", dest="enable_mgda", action="store_false")
    ap.set_defaults(enable_mgda=True)
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

    override = {}
    for key in (
        "lambda_robust", "lambda_tail", "lambda_diversity", "lambda_balance", "lambda_corr",
        "lambda_entropy", "lambda_uniformity", "lambda_consistency", "lambda_binary_collision",
        "topk_negatives", "binary_collision_target", "robustness_guard_batch_fraction",
        "robustness_guard_lambda_init", "robustness_guard_lambda_growth", "robustness_guard_lambda_max",
    ):
        arg = key
        if key == "robustness_guard_batch_fraction":
            arg = "guard_batch_fraction"
        elif key == "robustness_guard_lambda_init":
            arg = "guard_lambda"
        elif key == "robustness_guard_lambda_growth":
            arg = "guard_growth"
        elif key == "robustness_guard_lambda_max":
            arg = "guard_max"
        value = getattr(args, arg, None)
        if value is not None:
            override[key] = value
    override.update({
        "enable_selective_guard": args.enable_selective_guard,
        "enable_hard_negative_mining": args.enable_hard_negative_mining,
        "enable_memory_bank": args.enable_memory_bank,
        "enable_mgda": args.enable_mgda,
    })

    config = FinalCAPZWConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        nbits=args.bits,
        device=device,
        **override,
    )
    model = CAPZWHashNet(nbits=args.bits)
    history = train_cap_zw_v11(model, loader, config, checkpoint=args.checkpoint, history_path=args.history)
    print(
        f"Training complete: {args.experiment_name} version={CAP_ZW_FINAL} seed={args.seed} "
        f"images={len(images)} effective_samples={len(dataset)} bits={args.bits} "
        f"epochs={len(history)} device={device}"
    )
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
