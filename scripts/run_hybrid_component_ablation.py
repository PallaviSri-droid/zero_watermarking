from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    ("dino_only", 1.0, 0.0, 0.0),
    ("logpolar_only", 0.0, 1.0, 0.0),
    ("mrelbp_only", 0.0, 0.0, 1.0),
    ("dino_logpolar", 1.0, 1.0, 0.0),
    ("dino_mrelbp", 1.0, 0.0, 1.0),
    ("logpolar_mrelbp", 0.0, 1.0, 1.0),
    ("dino_logpolar_mrelbp", 1.0, 1.0, 1.0),
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered component ablation for the hybrid baseline.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="train_val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--fit-limit", type=int, default=1000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path("experiments") / "hybrid_component_ablation"
    root.mkdir(parents=True, exist_ok=True)
    for name, dw, lw, mw in VARIANTS:
        outdir = root / name
        summary = outdir / f"summary_seed_{args.seed}.csv"
        if args.skip_existing and summary.exists():
            print(f"Skipping {name}: output exists")
            continue
        command = [
            sys.executable,
            "scripts/benchmark_logpolar_dino_mrelbp.py",
            "--manifest", args.manifest,
            "--split", args.split,
            "--fit-split", args.fit_split,
            "--limit", str(args.limit),
            "--fit-limit", str(args.fit_limit),
            "--size", str(args.size),
            "--bits", str(args.bits),
            "--seed", str(args.seed),
            "--device", args.device,
            "--output", str(outdir),
            "--method-name", name,
            "--dino-weight", str(dw),
            "--logpolar-weight", str(lw),
            "--mrelbp-weight", str(mw),
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
    print(f"Hybrid component ablations: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
