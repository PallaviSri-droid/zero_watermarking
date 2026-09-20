from __future__ import annotations

"""DINOv2 adaptation of the DINOv3 + multi-scale neighborhood Zernike paper.

This module is a paper-faithful 2D dense-feature adaptation:
- frozen DINOv2 dense patch tokens,
- 3x3 global neighborhoods with center replacement,
- four overlapping 2x2 local sub-neighborhoods,
- complex Zernike moments up to configurable order,
- magnitude + channel L2 aggregation,
- mean-thresholded binary patch map.

The paper does not publish source code in the supplied material (data are
described as available on request), so this is an explicit reproduction
adaptation rather than a claim of exact source-code equivalence.
"""

from dataclasses import dataclass
from math import factorial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


PAPER_DINOV2_ZERNIKE_VERSION = "DINOv2-MNZM-v1"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _zernike_pairs(max_order: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                pairs.append((n, m))
    return pairs


def _zernike_basis(
    size: int, max_order: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    # The paper normalizes the polar radius by r/sqrt(2), putting square
    # corners on the unit circle.
    x = xx / np.sqrt(2.0)
    y = yy / np.sqrt(2.0)
    r = torch.sqrt(x * x + y * y).clamp_min(1e-12)
    theta = torch.atan2(y, x)
    mask = r <= 1.0 + 1e-7

    real_terms: list[Tensor] = []
    imag_terms: list[Tensor] = []
    for n, m in _zernike_pairs(max_order):
        radial = torch.zeros_like(r)
        ma = abs(m)
        for s in range((n - ma) // 2 + 1):
            coeff = ((-1) ** s) * factorial(n - s) / (
                factorial(s)
                * factorial((n + ma) // 2 - s)
                * factorial((n - ma) // 2 - s)
            )
            radial = radial + float(coeff) * r.pow(n - 2 * s)

        angle = m * theta
        real_terms.append(
            torch.where(mask, radial * torch.cos(angle), torch.zeros_like(r)).reshape(-1)
        )
        imag_terms.append(
            torch.where(mask, radial * torch.sin(angle), torch.zeros_like(r)).reshape(-1)
        )

    return torch.complex(torch.stack(real_terms), torch.stack(imag_terms))


@dataclass
class DinoV2MNZernikeConfig:
    model_name: str = "dinov2_vitb14_reg"
    input_size: int = 224
    patch_size: int = 14
    max_order: int = 8
    device: str = "auto"
    dense_batch_size: int = 4


class DINOv2MNZernike:
    """Frozen DINOv2 + multi-scale neighborhood Zernike extractor."""

    def __init__(self, config: Optional[DinoV2MNZernikeConfig] = None) -> None:
        self.config = config or DinoV2MNZernikeConfig()
        self.device = _resolve_device(self.config.device)
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.config.model_name)
        self.backbone.eval().to(self.device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.pairs = _zernike_pairs(self.config.max_order)
        self.zm_dim = len(self.pairs)
        self._basis_cache: dict[int, Tensor] = {}

    def _basis(self, size: int, dtype: torch.dtype = torch.float32) -> Tensor:
        basis = self._basis_cache.get(size)
        if basis is None:
            basis = _zernike_basis(size, self.config.max_order, self.device, dtype)
            self._basis_cache[size] = basis
        return basis

    @torch.inference_mode()
    def patch_tokens(self, x: Tensor) -> Tensor:
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (self.config.input_size, self.config.input_size):
            x = F.interpolate(
                x,
                size=(self.config.input_size, self.config.input_size),
                mode="bicubic",
                align_corners=False,
            )

        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        chunks: list[Tensor] = []

        for start in range(0, x.shape[0], self.config.dense_batch_size):
            xb = (x[start : start + self.config.dense_batch_size] - mean) / std
            features = self.backbone.forward_features(xb)
            tokens = features.get("x_norm_patchtokens")
            if tokens is None:
                tokens = features.get("x_prenorm")
                if tokens is None:
                    raise RuntimeError("DINOv2 output does not expose patch tokens")
                # Remove CLS and any register/storage tokens.
                expected = (self.config.input_size // self.config.patch_size) ** 2
                tokens = tokens[:, -expected:]
            chunks.append(tokens.float())

        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def _zm_from_windows(self, windows: Tensor, size: int) -> Tensor:
        # windows: [B, P, S*S, D]
        basis = self._basis(size, windows.dtype)
        real = torch.einsum("ks,bpsd->bpkd", basis.real, windows)
        imag = torch.einsum("ks,bpsd->bpkd", basis.imag, windows)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-12)

        # Paper: normalize/aggregate over the feature-channel dimension with p=2.
        return torch.linalg.vector_norm(magnitude, ord=2, dim=-1)

    @torch.inference_mode()
    def multi_scale_neighborhood_zernike(self, tokens: Tensor) -> Tensor:
        batch, patches, dim = tokens.shape
        side = int(round(patches ** 0.5))
        if side * side != patches:
            raise RuntimeError(f"Expected square patch grid, got {patches} tokens")

        grid = tokens.reshape(batch, side, side, dim).permute(0, 3, 1, 2)
        padded = F.pad(grid, (1, 1, 1, 1), mode="constant", value=1.0)

        # [B, P, 9, D]
        neighborhoods = (
            F.unfold(padded, kernel_size=3)
            .transpose(1, 2)
            .reshape(batch, patches, dim, 9)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        # Replace center with the mean of its remaining eight neighbors.
        neighbor_mean = torch.cat(
            [neighborhoods[:, :, :4], neighborhoods[:, :, 5:]], dim=2
        ).mean(dim=2)
        neighborhoods[:, :, 4] = neighbor_mean

        global_zm = self._zm_from_windows(neighborhoods, size=3)

        # Four overlapping 2x2 subwindows inside each 3x3 global neighborhood.
        n = neighborhoods.reshape(batch * patches, 3, 3, dim)
        local = torch.stack(
            [
                n[:, 0:2, 0:2],
                n[:, 0:2, 1:3],
                n[:, 1:3, 0:2],
                n[:, 1:3, 1:3],
            ],
            dim=1,
        )
        local = local.reshape(batch, patches * 4, 4, dim)
        local_zm = self._zm_from_windows(local, size=2)
        local_zm = local_zm.reshape(batch, patches, 4, self.zm_dim).mean(dim=2)

        return torch.cat([global_zm, local_zm], dim=-1)

    @torch.inference_mode()
    def descriptor(self, x: Tensor) -> Tensor:
        tokens = self.patch_tokens(x)
        return self.multi_scale_neighborhood_zernike(tokens)

    @torch.inference_mode()
    def patch_scores(self, x: Tensor) -> Tensor:
        descriptor = self.descriptor(x)
        return torch.linalg.vector_norm(descriptor, ord=2, dim=-1)

    @torch.inference_mode()
    def binary_feature(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        descriptor = self.descriptor(x)
        scores = torch.linalg.vector_norm(descriptor, ord=2, dim=-1)
        threshold = scores.mean(dim=1, keepdim=True)
        bits = (scores >= threshold).to(torch.uint8)

        side = int(round(bits.shape[1] ** 0.5))
        if side * side != bits.shape[1]:
            raise RuntimeError("Patch count cannot be reshaped to a square feature image")

        return bits.reshape(-1, side, side), scores, descriptor


__all__ = [
    "PAPER_DINOV2_ZERNIKE_VERSION",
    "DinoV2MNZernikeConfig",
    "DINOv2MNZernike",
]
