from __future__ import annotations

import cv2
import numpy as np
from scipy.fftpack import dct
from skimage.feature import canny


def dct2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


def _pool_low_frequency(coeff: np.ndarray, n: int) -> np.ndarray:
    h, w = coeff.shape
    values = []
    for s in range(2, h + w - 1):
        for i in range(max(1, s - w + 1), min(h - 1, s) + 1):
            j = s - i
            if 0 <= j < w and i < h:
                values.append(abs(coeff[i, j]))
            if len(values) >= n:
                return np.asarray(values[:n], dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    return np.pad(values, (0, max(0, n - len(values))))[:n]


def block_dct_signature(x: np.ndarray, hash_length: int = 256, balanced: bool = False) -> np.ndarray:
    coeff = dct2(x)
    vals = _pool_low_frequency(coeff, hash_length)
    threshold = float(np.median(vals) if balanced else vals.mean())
    return (vals >= threshold).astype(np.uint8)


def kaze_dct_signature(x: np.ndarray, hash_length: int = 256, balanced: bool = False) -> np.ndarray:
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    kaze = cv2.KAZE_create()
    _, descriptors = kaze.detectAndCompute(u8, None)
    if descriptors is None or len(descriptors) < 4:
        return block_dct_signature(x, hash_length, balanced)
    vector = descriptors.astype(np.float32).mean(axis=0)
    if len(vector) < hash_length:
        vector = np.resize(vector, hash_length)
    z = dct(vector[:hash_length], norm="ortho")
    threshold = float(np.median(z) if balanced else z.mean())
    return (z >= threshold).astype(np.uint8)


def edge_dct_signature(x: np.ndarray, hash_length: int = 256) -> np.ndarray:
    edges = canny(x, sigma=1.2).astype(np.float32)
    coeff = np.abs(dct2(edges)).ravel()
    coeff = np.pad(coeff, (0, max(0, hash_length - len(coeff))))[:hash_length]
    return (coeff >= np.median(coeff)).astype(np.uint8)


def random_projection_signature(x: np.ndarray, hash_length: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(hash_length, x.size)).astype(np.float32)
    scores = projection @ (x.ravel() - 0.5)
    return (scores >= 0).astype(np.uint8)


def method_registry(hash_length: int = 256):
    return {
        "DCT-Mean": lambda x: block_dct_signature(x, hash_length, False),
        "DCT-Balanced": lambda x: block_dct_signature(x, hash_length, True),
        "KAZE-DCT-Mean": lambda x: kaze_dct_signature(x, hash_length, False),
        "KAZE-DCT-Balanced": lambda x: kaze_dct_signature(x, hash_length, True),
        "Edge-DCT": lambda x: edge_dct_signature(x, hash_length),
    }
