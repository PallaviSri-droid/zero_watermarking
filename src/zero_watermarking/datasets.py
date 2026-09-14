from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    group_id: str
    modality: str = "unknown"
    label: str = "unknown"
    split: str = "unknown"


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """Load a CSV manifest with path/image_id/group_id columns.

    Empty/comment-only lines are ignored so the checked-in template can be
    edited safely. `group_id` should identify a patient/study when available.
    The optional `split` column is preserved because the publication benchmark
    must be able to filter the locked test partition deterministically.
    """
    frame = pd.read_csv(path, comment="#")
    required = {"path", "image_id", "group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["path", "image_id", "group_id"])
    records: list[ImageRecord] = []
    for row in frame.itertuples(index=False):
        records.append(
            ImageRecord(
                image_id=str(row.image_id),
                path=str(row.path),
                group_id=str(row.group_id),
                modality=str(getattr(row, "modality", "unknown")),
                label=str(getattr(row, "label", "unknown")),
                split=str(getattr(row, "split", "unknown")),
            )
        )
    return records


def validate_manifest(records: Iterable[ImageRecord], root: str | Path | None = None) -> pd.DataFrame:
    root_path = Path(root) if root else None
    rows = []
    for record in records:
        candidate = Path(record.path)
        if root_path and not candidate.is_absolute():
            candidate = root_path / candidate
        rows.append({
            "image_id": record.image_id,
            "path": str(candidate),
            "exists": candidate.exists(),
            "group_id": record.group_id,
            "modality": record.modality,
            "label": record.label,
            "split": record.split,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Manifest contains no records. Add real medical-image rows first.")
    if frame["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image_id values")
    if frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every record must have a non-empty group_id")
    if "split" in frame.columns:
        known = frame["split"].astype(str).str.strip()
        if known.eq("").any():
            raise ValueError("Every record must have a non-empty split value when split is present")
    return frame


def _group_shuffle_split_indices(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by group without importing scikit-learn.

    Unique groups are shuffled deterministically and accumulated until their
    row count is closest to the requested partition size. Whole groups are
    always kept together, preserving patient/study-level leakage protection.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    groups = frame["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two unique groups are required for splitting")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    group_sizes = {group: int(np.count_nonzero(groups == group)) for group in unique_groups}
    target_rows = max(1, int(round(len(frame) * test_size)))

    selected: list[str] = []
    selected_rows = 0
    remaining = list(shuffled)
    while remaining and (selected_rows < target_rows or not selected):
        best_index = min(
            range(len(remaining)),
            key=lambda i: abs(selected_rows + group_sizes[remaining[i]] - target_rows),
        )
        group = remaining.pop(best_index)
        selected.append(group)
        selected_rows += group_sizes[group]
        if selected_rows >= target_rows:
            break

    selected_set = set(selected)
    test_mask = np.array([group in selected_set for group in groups], dtype=bool)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Grouped split produced an empty partition")
    return train_idx, test_idx


def group_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0,1) and sum to < 1")

    train_val_idx, test_idx = _group_shuffle_split_indices(frame, test_size, seed)
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = _group_shuffle_split_indices(train_val, relative_val, seed + 1)
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    train_groups = set(train.group_id)
    val_groups = set(val.group_id)
    test_groups = set(test.group_id)
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("Grouped split leakage detected")
    return train, val, test


def load_image(path: str | Path, size: int = 224) -> np.ndarray:
    """Load a medical image as float32 grayscale in [0,1]."""
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values encountered in {path}")
    return np.clip(array, 0.0, 1.0)
