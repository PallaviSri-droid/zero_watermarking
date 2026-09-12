from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PATIENT_RE = re.compile(r"(?:patient|subject|case|study|participant)[_\- ]?([A-Za-z0-9]+)", re.I)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_") or "dataset"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_kaggle(dataset: str, output: Path, force: bool = False) -> Path:
    """Download a Kaggle dataset using the official Kaggle CLI."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".download_complete"
    if marker.exists() and not force:
        print(f"Using existing Kaggle download: {output}")
        return output
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output), "--unzip"]
    if force:
        cmd.append("--force")
    try:
        run(cmd)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Kaggle CLI is not installed. Run: pip install kaggle\n"
            "Then authenticate with: kaggle auth login"
        ) from exc
    marker.write_text("ok\n", encoding="utf-8")
    return output


def download_url(url: str, output: Path, force: bool = False) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    name = Path(url.split("?", 1)[0]).name or "download.bin"
    archive = output / name
    if archive.exists() and not force:
        print(f"Using existing download: {archive}")
    else:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, archive)

    if archive.suffix.lower() == ".zip":
        extract_dir = output / archive.stem
        if force and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        return extract_dir
    return output


def download_huggingface(dataset: str, output: Path, force: bool = False) -> Path:
    """Download a public Hugging Face dataset repository snapshot."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    if force and any(output.iterdir()):
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    snapshot_download(repo_id=dataset, repo_type="dataset", local_dir=str(output))
    return output


def infer_group_id(path: Path, root: Path) -> tuple[str, bool]:
    """Infer a group conservatively; mark inferred groups in the manifest."""
    relative_parts = path.relative_to(root).parts[:-1]
    text = "/".join(relative_parts)
    match = PATIENT_RE.search(text)
    if match:
        return match.group(0).lower().replace(" ", "_"), True
    if relative_parts:
        return relative_parts[-1], True
    return path.stem, True


def build_manifest(root: Path, manifest_path: Path, modality: str, label_from_parent: bool) -> int:
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No supported images found below {root}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "path", "group_id", "modality", "label", "group_inferred"])
        for index, image in enumerate(images):
            relative = image.relative_to(manifest_path.parents[2] if len(manifest_path.parents) >= 3 else Path.cwd())
            try:
                relative_str = relative.as_posix()
            except Exception:
                relative_str = image.as_posix()
            group_id, inferred = infer_group_id(image, root)
            label = image.parent.name if label_from_parent else "unknown"
            image_id = f"img_{index:06d}"
            writer.writerow([image_id, relative_str, group_id, modality, label, str(inferred).lower()])
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a public medical-image dataset and build a zero-watermarking manifest."
    )
    parser.add_argument("--source", choices=["kaggle", "huggingface", "url"], required=True)
    parser.add_argument("--dataset", required=True, help="Kaggle owner/name, HF dataset repo, or direct URL")
    parser.add_argument("--output", default="data/raw", help="Local dataset directory")
    parser.add_argument("--manifest", default="data/manifests/medical_manifest.csv")
    parser.add_argument("--modality", default="unknown")
    parser.add_argument("--label-from-parent", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.output).resolve()
    if args.source == "kaggle":
        dataset_root = download_kaggle(args.dataset, root / slug(args.dataset), args.force)
    elif args.source == "huggingface":
        dataset_root = download_huggingface(args.dataset, root / slug(args.dataset), args.force)
    else:
        dataset_root = download_url(args.dataset, root / "url_dataset", args.force)

    manifest = Path(args.manifest).resolve()
    count = build_manifest(dataset_root, manifest, args.modality, args.label_from_parent)
    print(f"Prepared {count} images")
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {manifest}")
    print("IMPORTANT: group_id is inferred unless your source provides patient/study metadata.")
    print("For publication-grade evaluation, replace inferred group_id values with verified patient/study IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
