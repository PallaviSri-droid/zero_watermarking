from __future__ import annotations

"""CAP + DINOv2 + Log-Polar adaptive fusion."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .learned import BEMQ, HashEncoder
from .logpolar_dino_mrelbp import logpolar_fourier_feature
from .training import _normalized_hamming

CAP_DINO_LP_VERSION = "CAP-DINO-LP-v4"


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
    quantizer_temperature: float = 2.0
    threshold_init_std: float = 0.05
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


class AdaptiveTriBranchGate(nn.Module):
    """Per-image mixture weights over CAP, DINOv2 and Log-Polar branches."""

    def __init__(self, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.temperature = float(max(temperature, 0.1))

    def forward(self, cap: Tensor, dino: Tensor, logpolar: Tensor) -> Tensor:
        logits = self.net(torch.cat([cap.detach(), dino.detach(), logpolar.detach()], dim=-1))
        return torch.softmax(logits / self.temperature, dim=-1)


class CAPDinoLogPolar(nn.Module):
    """Trainable CAP + frozen DINOv2 + deterministic Log-Polar fusion."""

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
        self.quantizer = BEMQ(
            self.config.bits,
            temperature=self.config.quantizer_temperature,
        )
        # Avoid the score == threshold tie at initialization. With the old
        # zero-threshold/high-temperature setup, p ~= 0.5 could coexist with
        # an all-one hard code and leave collision separation without useful
        # gradients at the exact equality point.
        nn.init.normal_(self.quantizer.thresholds, mean=0.0, std=self.config.threshold_init_std)

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
        x = F.interpolate(
            x,
            size=(self.config.dino_size, self.config.dino_size),
            mode="bicubic",
            align_corners=False,
        )
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        chunks: list[Tensor] = []
        for start in range(0, x.shape[0], 16):
            chunks.append(self.dino((x[start:start + 16] - mean) / std).float())
        return torch.cat(chunks, dim=0)

    def logpolar_features(self, x: Tensor) -> Tensor:
        images = x.detach().squeeze(1).cpu().numpy()
        features = np.stack([logpolar_fourier_feature(image, self.config.logpolar_size) for image in images])
        return torch.from_numpy(features).to(x.device, dtype=x.dtype)

    def branch_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        cap = F.normalize(self.cap_project(self.cap(x, hard=False)), dim=-1) * self.config.cap_weight
        dino = F.normalize(self.dino_project(self.dino_features(x)), dim=-1) * self.config.dino_weight
        logpolar = F.normalize(self.logpolar_project(self.logpolar_features(x)), dim=-1) * self.config.logpolar_weight
        gates = self.gate(cap, dino, logpolar)
        fused = gates[:, 0:1] * cap + gates[:, 1:2] * dino + gates[:, 2:3] * logpolar
        return cap, dino, logpolar, (gates, fused)

    def forward(self, x: Tensor, hard: bool = False) -> Tensor:
        _, _, _, (_, fused) = self.branch_features(x)
        logits = self.fusion(fused)
        return self.quantizer(logits, hard=hard)

    def forward_with_gate(self, x: Tensor, hard: bool = False) -> tuple[Tensor, Tensor]:
        _, _, _, (gates, fused) = self.branch_features(x)
        logits = self.fusion(fused)
        return self.quantizer(logits, hard=hard), gates


def _entropy_loss(codes: Tensor, target: float) -> Tensor:
    p = codes.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    return F.relu(float(target) - entropy).pow(2)


def _decorrelation_loss(codes: Tensor) -> Tensor:
    centered = codes - codes.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(2e-3)
    z = centered / std
    corr = (z.T @ z) / max(z.shape[0], 1)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    return ((corr - eye) * (1.0 - eye)).pow(2).mean()


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
    """Robustness + separation + binary collision-aware fusion objective."""
    robustness = (clean - attacked).abs().mean()

    labels2 = torch.cat([labels, labels], dim=0)
    codes = torch.cat([clean, attacked], dim=0)
    distances = _normalized_hamming(codes, codes)
    negatives = distances.masked_fill(labels2[:, None].eq(labels2[None, :]), float("inf"))
    finite = negatives[torch.isfinite(negatives)]
    if finite.numel():
        nearest = negatives.min(dim=1).values
        discrimination = F.softplus((margin - nearest) / 0.08).mul(0.08).mean()
        q05 = torch.quantile(finite, 0.05)
        q10 = torch.quantile(finite, 0.10)
        separation_tail = (
            F.softplus((q05_target - q05) / 0.025).mul(0.025)
            + 0.5 * F.softplus((q10_target - q10) / 0.025).mul(0.025)
        )
        binary_collision = F.relu(binary_target - finite).pow(2).mean()
    else:
        discrimination = clean.new_zeros(())
        separation_tail = clean.new_zeros(())
        binary_collision = clean.new_zeros(())

    entropy = _entropy_loss(codes, entropy_target)
    balance_p = codes.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    balance = (balance_p - 0.5).abs().mean()
    decorrelation = _decorrelation_loss(codes)

    mean_gate = gates.mean(dim=0)
    gate_balance = F.relu(0.20 - mean_gate).pow(2).mean()
    gate_entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(dim=1).mean()
    return {
        "robustness": robustness,
        "discrimination": discrimination,
        "separation_tail": separation_tail,
        "binary_collision": binary_collision,
        "entropy": entropy,
        "balance": balance,
        "decorrelation": decorrelation,
        "gate_balance": gate_balance,
        "gate_entropy": gate_entropy,
    }


def fusion_loss(terms: dict[str, Tensor]) -> Tensor:
    weights = {
        "robustness": 1.00,
        "discrimination": 0.85,
        "separation_tail": 1.15,
        "binary_collision": 0.95,
        "entropy": 0.90,
        "balance": 0.25,
        "decorrelation": 0.30,
        "gate_balance": 0.25,
        "gate_entropy": -0.05,
    }
    return sum(weights[name] * terms[name] for name in weights)


__all__ = [
    "CAP_DINO_LP_VERSION",
    "FusionConfig",
    "AdaptiveTriBranchGate",
    "CAPDinoLogPolar",
    "fusion_objective",
    "fusion_loss",
]
