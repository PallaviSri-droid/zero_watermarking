from __future__ import annotations

import hashlib
import numpy as np


def normalize_watermark(watermark, length: int) -> np.ndarray:
    if isinstance(watermark, str):
        raw = hashlib.sha256(watermark.encode("utf-8")).digest()
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    else:
        bits = np.asarray(watermark, dtype=np.uint8).ravel() & 1
    if bits.size < length:
        bits = np.resize(bits, length)
    return bits[:length].astype(np.uint8)


def create_zero_watermark(image_hash: np.ndarray, watermark, key: str = "", length: int = 256) -> np.ndarray:
    feature = np.asarray(image_hash, dtype=np.uint8).ravel() & 1
    feature = np.resize(feature, length)
    wm = normalize_watermark(watermark, length)
    key_bits = normalize_watermark(key, length) if key else np.zeros(length, dtype=np.uint8)
    return feature ^ wm ^ key_bits


def recover_watermark(image_hash: np.ndarray, zero_watermark: np.ndarray, key: str = "", length: int = 256) -> np.ndarray:
    feature = np.resize(np.asarray(image_hash, dtype=np.uint8).ravel() & 1, length)
    key_bits = normalize_watermark(key, length) if key else np.zeros(length, dtype=np.uint8)
    return feature ^ np.resize(np.asarray(zero_watermark, dtype=np.uint8).ravel() & 1, length) ^ key_bits


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel(); b = np.asarray(b, dtype=np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / d) if d else 1.0
