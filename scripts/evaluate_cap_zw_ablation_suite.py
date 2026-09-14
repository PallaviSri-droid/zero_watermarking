from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHOD_DIRS = (
    "full_cap_zw",
    "no_selective_guard",
    "no_binary_collision",
    "no_hard_negative_mining",
    "no_memory_bank",
    "fixed_weighted_sum",
    "robustness_only_reference",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate every retained CAP-ZW ablation on the locked test split.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default="experiments/ablations")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    root = Path(args.root)
    for name in METHOD_DIRS:
        checkpoint = root / name / "seed_42.pt"
        if not checkpoint.exists():
            print(f"Skipping {name}: checkpoint missing")
            continue
        out = root / name / "evaluation"
        cmd = [
            sys.executable,
            "scripts/evaluate_cap_zw.py",
            "--manifest", args.manifest,
            "--split", args.split,
            "--checkpoint", str(checkpoint),
            "--out", str(out),
            "--limit", str(args.limit),
            "--size", str(args.size),
            "--bits", str(args.bits),
            "--device", args.device,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Evaluations written under {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
