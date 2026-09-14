from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zero_watermarking.cap_dino_logpolar_v6 import CAPDinoLogPolarV6, V6Config, CAP_DINO_LP_V6_VERSION, v6_objective
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.protocol import seed_everything
from zero_watermarking.training import PairAttackDataset, _update_memory

ATTACKS = (
    "gaussian_noise", "salt_pepper", "gaussian_blur", "median_blur", "jpeg",
    "brightness", "contrast", "rotation", "crop_resize", "translation", "compound",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CAP-DINO-LogPolar v6 tail-robust fusion.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--attack-views", type=int, default=11)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_v6_seed42.pt")
    ap.add_argument("--history", default="experiments/results/cap_dino_logpolar_v6_seed42_training_history.csv")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 8:
        raise SystemExit("Need at least 8 existing images.")

    images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
    labels = np.arange(len(images), dtype=np.int64)
    dataset = PairAttackDataset(images, labels, attack_names=ATTACKS, attack_views=args.attack_views)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), generator=torch.Generator().manual_seed(args.seed),
    )

    cfg = V6Config(bits=args.bits, device=device, attack_views=args.attack_views)
    model = CAPDinoLogPolarV6(cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=2e-4)
    memory_codes = torch.empty((0, args.bits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        totals = {k: 0.0 for k in (
            "loss", "robustness", "robustness_q", "robustness_hard", "robustness_hard_tail",
            "branch_consistency", "gate_consistency", "attack_consistency", "discrimination", "tail",
            "binary_collision", "entropy", "balance", "decorrelation", "gate_usage", "gate_diversity",
            "observed_entropy", "observed_balance", "observed_robustness", "observed_hard_robustness",
            "observed_gate_std", "memory_size", "gate_0", "gate_1", "gate_2", "gate_entropy", "kept_cap",
            "kept_dino", "kept_logpolar",
        )}
        batches = 0
        for batch in loader:
            clean_x, attacked_x, labels_t, attack_names = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            pair = model.forward_pair(clean_x, attacked_x)
            mem = memory_codes if memory_codes.numel() else None
            mem_labels = memory_labels if memory_labels.numel() else None
            loss, terms = v6_objective(pair, labels_t, mem, mem_labels, cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            memory_codes, memory_labels = _update_memory(memory_codes, memory_labels, pair["clean_hard"].detach(), labels_t, cfg.memory_size)

            for k in totals:
                if k in terms:
                    totals[k] += float(loss.detach() if k == "loss" else terms[k].detach())
            totals["loss"] += float(loss.detach())
            g = pair["clean_gates"].detach()
            m = pair["branch_mask"].detach()
            totals["memory_size"] += float(memory_codes.shape[0])
            totals["gate_0"] += float(g[:, 0].mean())
            totals["gate_1"] += float(g[:, 1].mean())
            totals["gate_2"] += float(g[:, 2].mean())
            totals["gate_entropy"] += float((-(g.clamp_min(1e-8) * g.clamp_min(1e-8).log()).sum(dim=1)).mean())
            totals["kept_cap"] += float(m[:, 0].mean())
            totals["kept_dino"] += float(m[:, 1].mean())
            totals["kept_logpolar"] += float(m[:, 2].mean())
            batches += 1

        d = max(batches, 1)
        row = {"epoch": float(epoch + 1), **{k: v / d for k, v in totals.items()}}
        history.append(row)
        out = Path(args.checkpoint); out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": CAP_DINO_LP_V6_VERSION, "model": model.state_dict(), "config": cfg.__dict__,
                    "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "history": history}, out)
        hp = Path(args.history); hp.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(history).to_csv(hp, index=False)
        print(f"epoch={epoch+1} loss={row['loss']:.5f} soft_rob={row['robustness']:.5f} hard_rob={row['robustness_hard']:.5f} "
              f"entropy={row['observed_entropy']:.4f} gates={row['gate_0']:.3f}/{row['gate_1']:.3f}/{row['gate_2']:.3f} "
              f"gate_std={row['observed_gate_std']:.4f}")

    print(f"Training complete: {CAP_DINO_LP_V6_VERSION} seed={args.seed} images={len(images)} effective_samples={len(dataset)} bits={args.bits} epochs={args.epochs} device={device}")
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"history={Path(args.history).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
