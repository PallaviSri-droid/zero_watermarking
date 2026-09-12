from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from zero_watermarking.protocol import seed_everything


DEFAULT_SEEDS = (13, 23, 42, 73, 97)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the locked CAP-ZW candidate across independent seeds.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path("experiments")
    checkpoint_dir = root / "checkpoints" / "final_multiseed"
    result_dir = root / "results" / "final_multiseed"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_everything(seed)
        checkpoint = checkpoint_dir / f"seed_{seed}.pt"
        history = result_dir / f"seed_{seed}_training_history.csv"
        if args.skip_existing and checkpoint.exists() and history.exists():
            print(f"Skipping seed={seed}: outputs already exist")
            continue
        command = [
            sys.executable,
            "scripts/train_cap_zw_final.py",
            "--manifest", args.manifest,
            "--split", args.split,
            "--limit", str(args.limit),
            "--size", str(args.size),
            "--bits", str(args.bits),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
            "--device", args.device,
            "--checkpoint", str(checkpoint),
            "--history", str(history),
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

    print("All requested CAP-ZW seeds completed.")
    print(f"Checkpoints: {checkpoint_dir.resolve()}")
    print(f"Histories:   {result_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
