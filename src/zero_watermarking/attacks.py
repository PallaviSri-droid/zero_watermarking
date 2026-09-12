from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, median_filter, rotate, zoom, shift


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def gaussian_noise(x: np.ndarray, sigma: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _clip(x + rng.normal(0, sigma, x.shape))


def salt_pepper(x: np.ndarray, amount: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = x.copy()
    mask = rng.random(x.shape) < amount
    y[mask] = rng.integers(0, 2, size=int(mask.sum()))
    return y.astype(np.float32)


def blur(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return _clip(gaussian_filter(x, sigma))


def medblur(x: np.ndarray, size: int = 3) -> np.ndarray:
    return _clip(median_filter(x, size=size))


def jpeg(x: np.ndarray, quality: int = 70) -> np.ndarray:
    buf = io.BytesIO()
    image = Image.fromarray((_clip(x) * 255).astype(np.uint8))
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0


def brightness(x: np.ndarray, factor: float = 1.15) -> np.ndarray:
    image = Image.fromarray((_clip(x) * 255).astype(np.uint8))
    return np.asarray(ImageEnhance.Brightness(image).enhance(factor), dtype=np.float32) / 255.0


def contrast(x: np.ndarray, factor: float = 1.20) -> np.ndarray:
    image = Image.fromarray((_clip(x) * 255).astype(np.uint8))
    return np.asarray(ImageEnhance.Contrast(image).enhance(factor), dtype=np.float32) / 255.0


def rotate_image(x: np.ndarray, degrees: float = 5.0) -> np.ndarray:
    return _clip(rotate(x, degrees, reshape=False, order=1, mode="nearest"))


def crop_resize(x: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    h, w = x.shape
    dy, dx = int(h * fraction), int(w * fraction)
    if 2 * dy >= h or 2 * dx >= w:
        raise ValueError("crop fraction is too large for image size")
    y = x[dy : h - dy, dx : w - dx]
    return _clip(zoom(y, (h / y.shape[0], w / y.shape[1]), order=1))


def translation(x: np.ndarray, pixels: int = 4) -> np.ndarray:
    return _clip(shift(x, (pixels, pixels), order=1, mode="nearest"))


def compound(x: np.ndarray, seed: int = 0) -> np.ndarray:
    y = gaussian_noise(x, 0.03, seed)
    y = blur(y, 0.8)
    y = jpeg(y, 70)
    y = rotate_image(y, 3)
    return y


ATTACKS = {
    "clean": lambda x, **kw: x,
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper,
    "gaussian_blur": blur,
    "median_blur": medblur,
    "jpeg": jpeg,
    "brightness": brightness,
    "contrast": contrast,
    "rotation": rotate_image,
    "crop_resize": crop_resize,
    "translation": translation,
    "compound": compound,
}
