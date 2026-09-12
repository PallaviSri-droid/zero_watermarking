from __future__ import annotations

import numpy as np
from scipy.fft import dctn


def dct_low_frequency(image: np.ndarray, n: int = 256) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    c = dctn(x, type=2, norm="ortho")
    h, w = c.shape
    values: list[float] = []
    for s in range(h + w - 1):
        for i in range(max(0, s - (w - 1)), min(h - 1, s) + 1):
            j = s - i
            if i < h and j < w:
                values.append(float(abs(c[i, j])))
                if len(values) == n:
                    return np.asarray(values, dtype=np.float32)
    return np.pad(np.asarray(values, dtype=np.float32), (0, max(0, n - len(values))))[:n]


def dtcwt_features(image: np.ndarray, n: int = 256, levels: int = 3) -> np.ndarray:
    """DTCWT magnitude descriptor; uses DCT fallback only when dependency fails."""
    try:
        import dtcwt  # type: ignore
        x = np.asarray(image, dtype=np.float32)
        coeffs = dtcwt.Transform2d().forward(x, nlevels=levels)
        parts = [np.abs(np.asarray(c)).ravel() for c in coeffs.highpasses]
        parts.append(np.abs(np.asarray(coeffs.lowpass)).ravel())
        vector = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        return np.resize(vector.astype(np.float32), n) if vector.size else dct_low_frequency(x, n)
    except Exception:
        return dct_low_frequency(image, n)


def transform_binary_hash(image: np.ndarray, nbits: int = 256, transform: str = "dct") -> np.ndarray:
    values = dct_low_frequency(image, nbits) if transform.lower() == "dct" else dtcwt_features(image, nbits)
    threshold = float(np.median(values))
    return (values >= threshold).astype(np.uint8)
