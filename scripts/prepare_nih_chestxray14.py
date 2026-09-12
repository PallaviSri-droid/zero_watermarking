from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
KAGGLE_DATASET = "nih-chest-xrays/data"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_and_extract(root: Path, force: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "nih_chestxray14"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    marker = dataset_dir / ".kaggle_download_complete"

    if not marker.exists() or force:
        if force:
            for child in dataset_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        try:
            run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dataset_dir), "--force"])
        except FileNotFoundError as exc:
            raise SystemExit("Install Kaggle first: pip install kaggle, then run: kaggle auth login") from exc
        marker.write_text("ok\n", encoding="utf-8")

    # Kaggle may provide nested image ZIPs. Extract every ZIP recursively.
    changed = True
    while changed:
        changed = False
        for archive in sorted(dataset_dir.rglob("*.zip")):
            target = archive.with_suffix("")
            marker_file = target / ".extracted"
            if marker_file.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {archive.name} -> {target}")
            shutil.unpack_archive(str(archive), str(target))
            marker_file.write_text("ok\n", encoding="utf-8")
            changed = True
    return dataset_dir


def locate_metadata(root: Path) -> Path:
    candidates = list(root.rglob("Data_Entry_2017*.csv")) + list(root.rglob("data_entry_2017*.csv"))
    if not candidates:
        raise SystemExit("Could not find the NIH metadata CSV (Data_Entry_2017*.csv) after extraction.")
    return candidates[0]


def locate_split_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def load_official_split(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    train_file = locate_split_file(root, "train_val_list.txt")
    test_file = locate_split_file(root, "test_list.txt")
    if train_file:
        mapping.update({line.strip(): "train_val" for line in train_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()})
    if test_file:
        mapping.update({line.strip(): "test" for line in test_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()})
    return mapping


def build_manifest(root: Path, output: Path, max_images: int | None) -> int:
    metadata_path = locate_metadata(root)
    meta = pd.read_csv(metadata_path)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required.difference(meta.columns)
    if missing:
        raise SystemExit(f"NIH metadata is missing columns: {sorted(missing)}")

    meta["Image Index"] = meta["Image Index"].astype(str)
    meta["Patient ID"] = meta["Patient ID"].astype(str)
    official = load_official_split(root)

    image_map: dict[str, Path] = {}
    for image in root.rglob("*"):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            image_map.setdefault(image.name, image)

    rows: list[dict[str, str]] = []
    for row in meta.itertuples(index=False):
        filename = str(getattr(row, "Image_Index"))
        path = image_map.get(filename)
        if path is None:
            continue
        labels = str(getattr(row, "Finding_Labels"))
        split = official.get(filename, "unknown")
        rows.append(
            {
                "image_id": Path(filename).stem,
                "path": path.resolve().as_posix(),
                "group_id": str(getattr(row, "Patient_ID")),
                "modality": "chest_xray",
                "label": labels,
                "split": split,
                "view_position": str(getattr(row, "View_Position", "unknown")),
                "group_inferred": "false",
            }
        )
        if max_images and len(rows) >= max_images:
            break

    if not rows:
        raise SystemExit("No NIH images could be matched to the metadata CSV. Check extraction/download completeness.")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)

    counts = frame["split"].value_counts(dropna=False).to_dict()
    print(f"Manifest written: {output}")
    print(f"Matched images: {len(frame)}")
    print(f"Split counts: {counts}")
    if "unknown" in counts:
        print("WARNING: some images are not present in the standard train_val/test split files.")
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 and build a verified patient-level manifest.")
    parser.add_argument("--output", default="data/raw", help="Download/extraction directory")
    parser.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for a local pilot run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = download_and_extract(Path(args.output).resolve(), args.force)
    build_manifest(root, Path(args.manifest).resolve(), args.max_images)
    print("Ready for: python scripts/run_all.py --real --limit 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
