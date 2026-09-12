from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


class STEBinary(nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        soft = torch.sigmoid(logits)
        hard = (logits >= 0).to(logits.dtype)
        return hard + soft - soft.detach()


class BEMQ(nn.Module):
    """Balanced entropy-maximizing quantizer with learnable per-bit thresholds.

    Balance/entropy are objectives, not guarantees; experiments must verify them.
    """
    def __init__(self, nbits: int, temperature: float = 8.0):
        super().__init__()
        self.thresholds = nn.Parameter(torch.zeros(nbits))
        self.temperature = temperature

    def forward(self, scores: Tensor, hard: bool = True) -> Tensor:
        soft = torch.sigmoid(self.temperature * (scores - self.thresholds))
        if not hard:
            return soft
        hard_bits = (scores >= self.thresholds).to(scores.dtype)
        return hard_bits + soft - soft.detach()


class HashEncoder(nn.Module):
    def __init__(self, nbits: int = 256, base_channels: int = 32, use_bemq: bool = True):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, 2*c, 3, 2, 1), nn.BatchNorm2d(2*c), nn.GELU(),
            nn.Conv2d(2*c, 4*c, 3, 2, 1), nn.BatchNorm2d(4*c), nn.GELU(),
            nn.Conv2d(4*c, 8*c, 3, 2, 1), nn.BatchNorm2d(8*c), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(8*c, nbits)
        self.quantizer = BEMQ(nbits) if use_bemq else STEBinary()

    def forward(self, x: Tensor, hard: bool = True) -> Tensor:
        logits = self.head(self.features(x).flatten(1))
        return self.quantizer(logits, hard=hard) if isinstance(self.quantizer, BEMQ) else self.quantizer(logits)


@dataclass(frozen=True)
class LossWeights:
    robustness: float = 1.0
    discrimination: float = 1.0
    balance: float = 0.10
    decorrelation: float = 0.05
    entropy: float = 0.05


def pairwise_hamming(z: Tensor) -> Tensor:
    return torch.cdist(z, z, p=1) / z.shape[1]


def cap_zw_losses(clean: Tensor, attacked: Tensor, labels: Tensor, margin: float = 0.30) -> Dict[str, Tensor]:
    robustness = (clean - attacked).abs().mean()
    distances = pairwise_hamming(clean)
    same = labels[:, None].eq(labels[None, :])
    negatives = distances.masked_fill(same, float("inf")).min(1).values
    discrimination = torch.relu(margin - negatives).mean()
    p = clean.mean(0).clamp(1e-5, 1-1e-5)
    balance = (p - 0.5).pow(2).mean()
    entropy = -(p*torch.log2(p) + (1-p)*torch.log2(1-p)).mean()
    centered = clean - clean.mean(0, keepdim=True)
    cov = centered.T @ centered / max(clean.shape[0]-1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    decorrelation = off.pow(2).mean()
    return {"robustness": robustness, "discrimination": discrimination, "balance": balance, "decorrelation": decorrelation, "entropy_penalty": 1.0-entropy}


def weighted_cap_zw_loss(losses: Dict[str, Tensor], weights: LossWeights = LossWeights()) -> Tensor:
    return (weights.robustness*losses["robustness"] + weights.discrimination*losses["discrimination"] +
            weights.balance*losses["balance"] + weights.decorrelation*losses["decorrelation"] +
            weights.entropy*losses["entropy_penalty"])
