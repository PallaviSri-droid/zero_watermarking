from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


class STEBinary(nn.Module):
    """Straight-through binary quantizer: hard forward, sigmoid gradient."""

    def forward(self, logits: Tensor) -> Tensor:
        soft = torch.sigmoid(logits)
        hard = (logits >= 0).to(logits.dtype)
        return hard + soft - soft.detach()


class HashEncoder(nn.Module):
    """Compact single-channel encoder for zero-watermark experiments."""

    def __init__(self, nbits: int = 256, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, 2, 1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1),
            nn.BatchNorm2d(2 * c),
            nn.GELU(),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1),
            nn.BatchNorm2d(4 * c),
            nn.GELU(),
            nn.Conv2d(4 * c, 8 * c, 3, 2, 1),
            nn.BatchNorm2d(8 * c),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(8 * c, nbits)
        self.quantizer = STEBinary()

    def forward(self, x: Tensor, hard: bool = True) -> Tensor:
        logits = self.head(self.features(x).flatten(1))
        return self.quantizer(logits) if hard else logits


@dataclass(frozen=True)
class LossWeights:
    robustness: float = 1.0
    discrimination: float = 1.0
    balance: float = 0.1
    decorrelation: float = 0.05


def _pairwise_hamming(z: Tensor) -> Tensor:
    return torch.cdist(z, z, p=1) / z.shape[1]


def cap_zw_losses(
    clean_bits: Tensor,
    attack_bits: Tensor,
    labels: Tensor,
    margin: float = 0.35,
) -> Dict[str, Tensor]:
    """Return differentiable CAP-ZW objectives.

    - robustness: same-image clean/attack agreement
    - discrimination: hard-negative margin on different images
    - balance: each bit approaches 0.5
    - decorrelation: off-diagonal code covariance is penalized
    """
    robustness = (clean_bits - attack_bits).abs().mean()

    distances = _pairwise_hamming(clean_bits)
    same = labels[:, None].eq(labels[None, :])
    negative_distances = distances.masked_fill(same, float("inf"))
    hard_negative = negative_distances.min(dim=1).values
    discrimination = torch.relu(margin - hard_negative).mean()

    balance = (clean_bits.mean(dim=0) - 0.5).pow(2).mean()

    centered = clean_bits - clean_bits.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(clean_bits.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    decorrelation = off_diagonal.pow(2).mean()

    return {
        "robustness": robustness,
        "discrimination": discrimination,
        "balance": balance,
        "decorrelation": decorrelation,
    }


def weighted_cap_zw_loss(losses: Dict[str, Tensor], weights: LossWeights = LossWeights()) -> Tensor:
    return (
        weights.robustness * losses["robustness"]
        + weights.discrimination * losses["discrimination"]
        + weights.balance * losses["balance"]
        + weights.decorrelation * losses["decorrelation"]
    )
