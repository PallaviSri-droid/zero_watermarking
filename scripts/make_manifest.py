from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a zero-watermarking dataset manifest")
    parser.add_argument("root", type=Path, help="Root folder containing medical images")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument(
        "--group-regex",
        default=r"(?P<group>[^_]+)",
        help="Regex applied to the filename stem; named group 'group' becomes group_id",
    )
    parser.add_argument("--modality", default="unknown")
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)
    pattern = re.compile(args.group_regex)
    rows = []
    for path in sorted(args.root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        match = pattern.search(path.stem)
        group_id = match.group("group") if match and "group" in match.groupdict() else path.stem
        rows.append(
            {
                "image_id": path.stem,
                "path": str(path.resolve()),
                "group_id": str(group_id),
                "modality": args.modality,
                "label": "unknown",
            }
        )

    if not rows:
        raise RuntimeError(f"No supported images found in {args.root}")
    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} records to {args.output}")


if __name__ == "__main__":
    main()
