from __future__ import annotations

"""LogPolar + DINOv2 + MRELBP deterministic zero-watermarking baseline.

This module implements an independently reproducible fusion baseline rather than
copying code from any paper.  It combines three established descriptor families:

1. DINOv2 global visual features (frozen pretrained ViT-S/14).
2. Log-polar Fourier-magnitude features, which convert rotation/scale variation
   into shifts before taking a shift-insensitive magnitude spectrum.
3. Median Robust Extended Local Binary Pattern (MRELBP)-style multiscale median
   histograms for noise/texture robustness.

The fused descriptor is converted to a fixed-length binary signature using a
fitted PCA + sign projection.  PCA is fitted only on the training/validation
reference images supplied to ``fit`` and must never be fitted on the locked test
set.

Research note: DINOv2, Log-Polar descriptors and MRELBP are established building
blocks.  This file is a benchmark implementation of their fusion, not a claim
that the fusion itself is previously published or novel.
"""

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class HybridConfig:
    bits: int = 128
    dino_model: str = "dinov2_vits14"
    image_size: int = 224
    logpolar_size: int = 128
    mrelbp_radius: int = 2
    mrelbp_points: int = 8
    mrelbp_radii: tuple[int, ...] = (1, 2, 4)
    pca_components: int = 256
    dino_weight: float = 1.0
    logpolar_weight: float = 1.0
    mrelbp_weight: float = 1.0
    device: str = "auto"
    seed: int = 42


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torch.device(name)
    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return out


def logpolar_fourier_feature(image: np.ndarray, size: int = 128) -> np.ndarray:
    """Compact rotation/scale-oriented log-polar Fourier magnitude descriptor."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("logpolar descriptor expects a grayscale 2-D image")
    h, w = x.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = max(2.0, min(cx, cy))
    warped = cv2.warpPolar(
        x,
        (size, size),
        (cx, cy),
        radius,
        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR,
    )
    fft = np.fft.fft2(warped)
    mag = np.log1p(np.abs(fft)).astype(np.float32)
    # Remove DC dominance and retain a compact fixed-size representation.
    mag[0, 0] = 0.0
    small = cv2.resize(mag, (32, 32), interpolation=cv2.INTER_AREA)
    vec = small.reshape(-1)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-8)


def _bilinear_sample(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = image.shape
    ys0 = np.floor(ys).astype(np.int32)
    xs0 = np.floor(xs).astype(np.int32)
    ys1 = np.clip(ys0 + 1, 0, h - 1)
    xs1 = np.clip(xs0 + 1, 0, w - 1)
    ys0 = np.clip(ys0, 0, h - 1)
    xs0 = np.clip(xs0, 0, w - 1)
    wy = ys - ys0
    wx = xs - xs0
    return (
        (1 - wy) * (1 - wx) * image[ys0, xs0]
        + (1 - wy) * wx * image[ys0, xs1]
        + wy * (1 - wx) * image[ys1, xs0]
        + wy * wx * image[ys1, xs1]
    )


def _median_map(image: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.medianBlur(np.asarray(image, dtype=np.float32), k)


def _lbp_hist(code: np.ndarray, bins: int) -> np.ndarray:
    hist = np.bincount(code.ravel().astype(np.int64), minlength=bins).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return hist


def mrelbp_feature(
    image: np.ndarray,
    radii: Iterable[int] = (1, 2, 4),
    points: int = 8,
) -> np.ndarray:
    """Practical MRELBP-style multiscale median local-pattern histogram.

    The implementation follows the defining idea of MRELBP: compare regional
    medians rather than raw pixel intensities, at multiple spatial scales, and
    concatenate normalized local-pattern histograms.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("MRELBP expects grayscale 2-D images")
    parts: list[np.ndarray] = []
    for radius in tuple(radii):
        center = _median_map(x, radius)
        angles = 2.0 * np.pi * np.arange(points, dtype=np.float32) / points
        ys = np.arange(x.shape[0], dtype=np.float32)[:, None]
        xs = np.arange(x.shape[1], dtype=np.float32)[None, :]
        cy = ys + radius * np.sin(angles)[:, None, None]
        cx = xs + radius * np.cos(angles)[:, None, None]
        samples = np.stack(
            [_bilinear_sample(x, cy[i], cx[i]) for i in range(points)], axis=0
        )
        code = np.zeros(x.shape, dtype=np.uint16)
        center_value = center[None, :, :]
        for i in range(points):
            local_median = _median_map(samples[i], max(1, radius // 2))
            code |= ((local_median >= center_value[0]).astype(np.uint16) << i)
        parts.append(_lbp_hist(code, 2**points))

    # Add a low-cost center/neighbor median relation histogram at each scale.
    for radius in tuple(radii):
        center = _median_map(x, radius)
        ring = cv2.blur(x, (2 * radius + 1, 2 * radius + 1))
        signed = (ring >= center).astype(np.uint8)
        parts.append(_lbp_hist(signed, 2))
    return np.concatenate(parts).astype(np.float32)


class LogPolarDinoMRELBP:
    """Fitted hybrid descriptor + deterministic binary hash encoder."""

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self.device = _device(self.config.device)
        self.dino = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.projection: np.ndarray | None = None
        self._load_dino()

    def _load_dino(self) -> None:
        # Official Meta DINOv2 weights are loaded through PyTorch Hub.
        self.dino = torch.hub.load("facebookresearch/dinov2", self.config.dino_model)
        self.dino.eval().to(self.device)
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _dino_features(self, images: np.ndarray) -> np.ndarray:
        # Input images are grayscale [N,H,W] in [0,1]. DINO expects 3 channels.
        x = torch.from_numpy(images.astype(np.float32))[:, None, :, :]
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.config.image_size, self.config.image_size), mode="bicubic", align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
        x = (x.to(self.device) - mean) / std
        outputs: list[np.ndarray] = []
        for start in range(0, len(x), 16):
            feat = self.dino(x[start : start + 16])
            outputs.append(feat.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _raw_features(self, images: np.ndarray) -> np.ndarray:
        dino = self._dino_features(images)
        logpolar = np.stack([logpolar_fourier_feature(x, self.config.logpolar_size) for x in images])
        mrelbp = np.stack([
            mrelbp_feature(x, self.config.mrelbp_radii, self.config.mrelbp_points)
            for x in images
        ])
        blocks = []
        for block, weight in (
            (dino, self.config.dino_weight),
            (logpolar, self.config.logpolar_weight),
            (mrelbp, self.config.mrelbp_weight),
        ):
            block = block.astype(np.float32)
            block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-8)
            blocks.append(block * float(weight))
        return np.concatenate(blocks, axis=1)

    def fit(self, images: np.ndarray) -> "LogPolarDinoMRELBP":
        raw = self._raw_features(images)
        self.scaler = StandardScaler().fit(raw)
        scaled = self.scaler.transform(raw)
        n_components = min(self.config.pca_components, scaled.shape[0], scaled.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.config.seed).fit(scaled)
        projected = self.pca.transform(scaled)
        rng = np.random.default_rng(self.config.seed)
        self.projection = rng.normal(size=(projected.shape[1], self.config.bits)).astype(np.float32)
        self.projection /= np.maximum(np.linalg.norm(self.projection, axis=0, keepdims=True), 1e-8)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None or self.projection is None:
            raise RuntimeError("fit() must be called before transform()")
        raw = self._raw_features(images)
        projected = self.pca.transform(self.scaler.transform(raw))
        scores = projected @ self.projection
        return (scores >= 0).astype(np.uint8)


__all__ = ["HybridConfig", "LogPolarDinoMRELBP", "logpolar_fourier_feature", "mrelbp_feature"]
