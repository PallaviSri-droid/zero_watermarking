from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete reproducibility pipeline.")
    parser.add_argument("--real", action="store_true", help="Run the real medical benchmark; requires a populated manifest.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of real images.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    smoke = [sys.executable, str(project_root / "scripts" / "run_smoke_test.py")]
    print("[1/2] Running offline smoke test...")
    subprocess.run(smoke, cwd=project_root, check=True)

    if args.real:
        manifest = project_root / "data" / "manifests" / "medical_manifest.csv"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest: {manifest}")
        print("[2/2] Running real medical benchmark...")
        cmd = [sys.executable, str(project_root / "scripts" / "run_full_benchmark.py"), "--manifest", str(manifest), "--limit", str(args.limit)]
        subprocess.run(cmd, cwd=project_root, check=True)
    else:
        print("[2/2] Real benchmark skipped. Add real rows to data/manifests/medical_manifest.csv and rerun with --real.")
    print("PIPELINE COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
