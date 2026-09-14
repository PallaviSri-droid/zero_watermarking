from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


BOOL_FLAGS = {
    "enable_selective_guard": "--no-selective-guard",
    "enable_hard_negative_mining": "--no-hard-negative-mining",
    "enable_memory_bank": "--no-memory-bank",
    "enable_mgda": "--no-mgda",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered CAP-ZW ablation suite.")
    ap.add_argument("--config", default="configs/cap_zw_ablations.yaml")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiments = cfg.get("experiments", [])
    root = Path("experiments") / "ablations"
    root.mkdir(parents=True, exist_ok=True)

    for item in experiments:
        name = str(item["name"])
        outdir = root / name
        checkpoint = outdir / f"seed_{args.seed}.pt"
        history = outdir / f"seed_{args.seed}_history.csv"
        if args.skip_existing and checkpoint.exists() and history.exists():
            print(f"Skipping {name}: outputs exist")
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
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint", str(checkpoint),
            "--history", str(history),
            "--experiment-name", name,
        ]
        for key, value in item.items():
            if key == "name":
                continue
            if key in BOOL_FLAGS:
                if value is False:
                    command.append(BOOL_FLAGS[key])
                continue
            flag = "--" + key.replace("_", "-")
            command.extend([flag, str(value)])

        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        (outdir / "ablation_spec.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    print(f"Ablation outputs: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
