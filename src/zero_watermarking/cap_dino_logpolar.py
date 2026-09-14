from __future__ import annotations

"""CAP + DINOv2 + Log-Polar adaptive fusion.

This candidate uses three complementary branches:
- CAP: trainable medical-image representation.
- DINOv2: frozen global visual representation.
- Log-polar Fourier magnitude: deterministic geometric cue.

The branches are projected to a shared space and fused with a per-image learned
soft gate. Training uses a collision-aware objective over binary straight-
through codes plus continuous robustness, lower-tail separation, entropy,
balance, decorrelation, and gate-collapse regularization.

The combination is a research candidate; the individual components are
established building blocks and are not claimed as novel individually.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .learned import BEMQ, HashEncoder
from .logpolar_dino_mrelbp import logpolar_fourier_feature
from .training import _normalized_hamming

CAP_DINO_LP_VERSION = "CAP-DINO-LP-v2"


@dataclass
class FusionConfig:
    bits: int = 128
    cap_dim: int = 192
    dino_dim: int = 384
    fusion_dim: int = 192
    dino_model: str = "dinov2_vits14"
    dino_size: int = 224
    logpolar_size: int = 128
    cap_weight: float = 1.0
    dino_weight: float = 1.0
    logpolar_weight: float = 0.85
    gate_temperature: float = 1.0
    dropout: float = 0.05
    device: str = "auto"
    freeze_dino: bool = True


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _soft_bit(z: Tensor) -> Tensor:
    return torch.sigmoid(z)


def _ste_bits(logits: Tensor) -> Tensor:
    soft = _soft_bit(logits)
    hard = (soft >= 0.5).to(soft.dtype)
    return hard.detach() - soft.detach() + soft


class AdaptiveTriBranchGate(nn.Module):
    """Per-sample mixture weights over CAP, DINO and Log-Polar branches."""

    def __init__(self, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.temperature = float(max(temperature, 0.1))

    def forward(self, cap: Tensor, dino: Tensor, logpolar: Tensor) -> Tensor:
        logits = self.net(torch.cat([cap, dino, logpolar], dim=-1))
        return torch.softmax(logits / self.temperature, dim=-1)


class CAPDinoLogPolar(nn.Module):
    """Trainable CAP branch fused with frozen DINOv2 and Log-Polar features."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionConfig()
        self.runtime_device = _resolve_device(self.config.device)

        self.cap = HashEncoder(nbits=self.config.cap_dim, base_channels=32, use_bemq=False)
        self.dino: nn.Module = torch.hub.load("facebookresearch/dinov2", self.config.dino_model)
        self.dino.eval()
        if self.config.freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad_(False)

        self.cap_project = nn.Sequential(nn.Linear(self.config.cap_dim, self.config.fusion_dim), nn.GELU())
        self.dino_project = nn.Sequential(nn.Linear(self.config.dino_dim, self.config.fusion_dim), nn.GELU())
        self.logpolar_project = nn.Sequential(nn.Linear(32 * 32, self.config.fusion_dim), nn.GELU())
        self.gate = AdaptiveTriBranchGate(self.config.fusion_dim, self.config.gate_temperature)

        self.fusion = nn.Sequential(
            nn.LayerNorm(self.config.fusion_dim),
            nn.Linear(self.config.fusion_dim, self.config.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fusion_dim, self.config.bits),
        )
        self.quantizer = BEMQ(self.config.bits, temperature=8.0)
        self.to(self.runtime_device)
        self.dino.to(self.runtime_device)

    def train(self, mode: bool = True) -> "CAPDinoLogPolar":
        super().train(mode)
        self.dino.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.runtime_device

    @torch.no_grad()
    def dino_features(self, x: Tensor) -> Tensor:
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.config.dino_size, self.config.dino_size), mode="bicubic", align_corners=False)
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.dino((x - mean) / std).float()

    def logpolar_features(self, x: Tensor) -> Tensor:
        images = x.detach().squeeze(1).cpu().numpy()
        feats = np.stack([logpolar_fourier_feature(img, self.config.logpolar_size) for img in images])
        return torch.from_numpy(feats).to(x.device, dtype=x.dtype)

    def branch_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cap = F.normalize(self.cap_project(self.cap(x, hard=False)), dim=-1) * self.config.cap_weight
        dino = F.normalize(self.dino_project(self.dino_features(x)), dim=-1) * self.config.dino_weight
        logpolar = F.normalize(self.logpolar_project(self.logpolar_features(x)), dim=-1) * self.config.logpolar_weight
        gates = self.gate(cap, dino, logpolar)
        fused = gates[:, 0:1] * cap + gates[:, 1:2] * dino + gates[:, 2:3] * logpolar
        return cap, dino, logpolar, (gates, fused)

    def forward(self, x: Tensor, hard: bool = False) -> Tensor:
        _, _, _, (gates, fused) = self.branch_features(x)
        logits = self.fusion(fused)
        return self.quantizer(logits, hard=hard)

    def forward_with_gate(self, x: Tensor, hard: bool = False) -> tuple[Tensor, Tensor]:
        _, _, _, (gates, fused) = self.branch_features(x)
        logits = self.fusion(fused)
        return self.quantizer(logits, hard=hard), gates


def fusion_objective(
    clean: Tensor,
    attacked: Tensor,
    labels: Tensor,
    gates: Tensor,
    margin: float = 0.36,
    q05_target: float = 0.10,
    q10_target: float = 0.15,
    entropy_target: float = 0.80,
    binary_target: float = 0.14,
) -> dict[str, Tensor]:
    """Collision-aware multi-objective terms for adaptive fusion."""
    robustness = (clean - attacked).abs().mean()
    codes = torch.cat([clean, attacked], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    hard = _ste_bits(codes)
    distances = _normalized_hamming(hard, hard)
    negatives = distances.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    finite = negatives[torch.isfinite(negatives)]

    nearest = negatives.min(dim=1).values
    discrimination = F.softplus((float(margin) - nearest) / 0.08).mul(0.08).mean()
    q05 = torch.quantile(finite, 0.05) if finite.numel() else clean.new_zeros(())
    q10 = torch.quantile(finite, 0.10) if finite.numel() else clean.new_zeros(())
    separation_tail = (
        F.softplus((float(q05_target) - q05) / 0.025).mul(0.025)
        + 0.5 * F.softplus((float(q10_target) - q10) / 0.025).mul(0.025)
    )

    p = hard.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    entropy_loss = F.relu(float(entropy_target) - entropy).pow(2)

    centered = hard - hard.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(2e-3)
    z = centered / std
    corr_mat = (z.T @ z) / max(hard.shape[0], 1)
    eye = torch.eye(corr_mat.shape[0], device=hard.device, dtype=hard.dtype)
    decorrelation = ((corr_mat - eye) * (1 - eye)).pow(2).mean()

    binary_collision = F.relu(float(binary_target) - negatives[torch.isfinite(negatives)]).pow(2).mean() if finite.numel() else clean.new_zeros(())

    gate_entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(dim=1).mean()
    gate_balance = F.relu(0.20 - gates.mean(dim=0)).pow(2).mean()

    return {
        "robustness": robustness,
        "discrimination": discrimination,
        "separation_tail": separation_tail,
        "entropy_loss": entropy_loss,
        "balance": balance,
        "decorrelation": decorrelation,
        "binary_collision": binary_collision,
        "gate_entropy": gate_entropy,
        "gate_balance": gate_balance,
        "q05": q05.detach(),
        "q10": q10.detach(),
        "bit_entropy": entropy.detach(),
    }


def fusion_loss(terms: dict[str, Tensor]) -> Tensor:
    """Conservative scalarization; final model selection remains benchmark-locked."""
    weights = {
        "robustness": 1.00,
        "discrimination": 0.80,
        "separation_tail": 1.10,
        "entropy_loss": 1.00,
        "balance": 0.25,
        "decorrelation": 0.30,
        "binary_collision": 0.90,
        "gate_entropy": -0.06,
        "gate_balance": 0.30,
    }
    return sum(weights[k] * terms[k] for k in weights)


__all__ = ["CAP_DINO_LP_VERSION", "FusionConfig", "AdaptiveTriBranchGate", "CAPDinoLogPolar", "fusion_objective", "fusion_loss"]
