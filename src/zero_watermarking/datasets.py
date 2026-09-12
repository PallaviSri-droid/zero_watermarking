from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    group_id: str
    modality: str = "unknown"
    label: str = "unknown"


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """Load a CSV manifest with path/image_id/group_id columns.

    Empty/comment-only lines are ignored so the checked-in template can be
    edited safely. `group_id` should identify a patient/study when available.
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
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Manifest contains no records. Add real image rows first.")
    if frame["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image_id values")
    if frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every record must have a non-empty group_id")
    return frame


def group_split(frame: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0,1) and sum to < 1")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(frame, groups=frame["group_id"]))
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    relative_val = val_size / (1.0 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_idx, val_idx = next(splitter2.split(train_val, groups=train_val["group_id"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    if set(train.group_id) & set(val.group_id) or set(train.group_id) & set(test.group_id) or set(val.group_id) & set(test.group_id):
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
