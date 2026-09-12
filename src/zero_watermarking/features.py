from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn


def normalize_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x, dtype=np.float32) if hi <= lo else (x - lo) / (hi - lo)


def kaze_descriptor(image: np.ndarray, nbits: int = 256, max_features: int = 512) -> np.ndarray:
    x = (normalize_image(image) * 255).astype(np.uint8)
    detector = cv2.KAZE_create()
    keypoints, desc = detector.detectAndCompute(x, None)
    if desc is None or len(desc) == 0:
        vec = x.reshape(-1).astype(np.float32)
    else:
        # Deterministic ordering by response, then location.
        order = sorted(range(len(keypoints)), key=lambda i: (-keypoints[i].response, keypoints[i].pt[1], keypoints[i].pt[0]))[:max_features]
        vec = desc[order].astype(np.float32).ravel()
    return dct_hash(vec, nbits)


def dct_features(image: np.ndarray, grid: int = 32) -> np.ndarray:
    x = normalize_image(image, grid)
    coeff = dctn(x, type=2, norm="ortho")
    return coeff[: grid // 2, : grid // 2].ravel()


def dct_hash(vector: np.ndarray, nbits: int = 256) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).ravel()
    if v.size == 0:
        return np.zeros(nbits, dtype=np.uint8)
    if v.size < nbits:
        v = np.resize(v, nbits)
    else:
        v = v[:nbits]
    threshold = float(np.median(v))
    return (v >= threshold).astype(np.uint8)


def transform_hash(image: np.ndarray, nbits: int = 256, transform: str = "dct") -> np.ndarray:
    if transform.lower() == "kaze":
        return kaze_descriptor(image, nbits)
    return dct_hash(dct_features(image), nbits)
