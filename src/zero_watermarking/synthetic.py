from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def make_medical_like(index: int, size: int = 224) -> np.ndarray:
    """Deterministic phantom-like image for pipeline validation only."""
    rng = np.random.default_rng(index)
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    image = np.zeros((size, size), dtype=np.float32)
    n = 4 + index % 5
    for _ in range(n):
        cx, cy = rng.uniform(-0.65, 0.65, 2)
        sx, sy = rng.uniform(0.08, 0.32, 2)
        amp = rng.uniform(0.3, 1.0)
        image += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    ring = np.exp(-((np.sqrt(xx**2 + yy**2) - 0.55) ** 2) / (2 * 0.02**2))
    image += 0.25 * ring
    image += 0.05 * rng.normal(size=(size, size))
    image = gaussian_filter(image, 0.8 + (index % 3) * 0.2)
    image -= image.min()
    image /= max(float(image.max()), 1e-8)
    return image.astype(np.float32)


def make_dataset(n: int = 40, size: int = 224) -> dict[int, np.ndarray]:
    return {i: make_medical_like(i, size) for i in range(n)}
