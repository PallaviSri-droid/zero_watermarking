from __future__ import annotations

import numpy as np
from scipy.fft import dctn


def dct_low_frequency(image: np.ndarray, n: int = 256) -> np.ndarray:
    x = np.asarray(image, np.float32)
    c = dctn(x, type=2, norm="ortho")
    block = c[: min(32, c.shape[0]), : min(32, c.shape[1])].ravel()
    return np.resize(block, n)


def dtcwt_low_frequency(image: np.ndarray, n: int = 256) -> np.ndarray:
    """Return a deterministic DTCWT low-pass descriptor when dtcwt is installed."""
    try:
        import dtcwt
    except ImportError as exc:
        raise ImportError("Install optional dependency 'dtcwt' to use DTCWT") from exc
    x = np.asarray(image, np.float32)
    transform = dtcwt.Transform2d()
    coeffs = transform.forward(x, nlevels=3)
    low = np.asarray(coeffs.lowpass, np.float32)
    return np.resize(low.ravel(), n)
