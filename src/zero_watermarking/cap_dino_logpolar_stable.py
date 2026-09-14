from __future__ import annotations

"""Evidence-backed stable CAP-DINO-LogPolar repair.

This is intentionally not a new architecture family. It preserves the v6 three-
branch fusion path and repairs the observed optimization failure modes by:
- warm-starting from the known-good v6 representation,
- using a smooth Bernoulli bit-flip surrogate for robustness,
- adding a standard representation-space contrastive objective,
- adding explicit quantization pressure,
- delaying strong collision pressure until the representation is stable,
- keeping gate regularization weak instead of forcing specialization.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .cap_dino_logpolar_v6 import CAPDinoLogPolarV6, V6Config, _finite, _safe_cosine_loss
from .training import _normalized_hamming

STABLE_VERSION = "CAP-DINO-LogPolar-stable-repair"


@dataclass
class StableConfig(V6Config):
    lr: float = 2e-5
    attack_views: int = 6
    robust_target: float = 0.065
    robust_q_target: float = 0.090
    robust_hard_target: float = 0.0
    q05_target: float = 0.10
    q10_target: float = 0.16
    entropy_target: float = 0.82
    balance_target: float = 0.12
    binary_collision_target: float = 0.16
    topk_negatives: int = 24
    memory_size: int = 4096
    quantizer_temperature: float = 1.5
    threshold_init_std: float = 0.08
    lambda_robust: float = 0.85
    lambda_robust_q: float = 0.35
    lambda_contrastive: float = 0.45
    lambda_branch_consistency: float = 0.20
    lambda_discrimination: float = 0.55
    lambda_tail: float = 0.70
    lambda_binary_collision: float = 0.65
    lambda_entropy: float = 0.25
    lambda_balance: float = 0.15
    lambda_decorrelation: float = 0.10
    lambda_quantization: float = 0.20
    lambda_gate_usage: float = 0.03
    stage2_start: float = 0.35
    contrastive_temperature: float = 0.10


class CAPDinoLogPolarStable(CAPDinoLogPolarV6):
    """V6 fusion path with conservative, differentiable objectives."""

    def __init__(self, config: StableConfig | None = None) -> None:
        super().__init__(config or StableConfig())

    def forward_pair(self, clean_x: Tensor, attacked_x: Tensor):
        clean_cap, clean_dino, clean_lp = self.encode_branches(clean_x)
        attacked_cap, attacked_dino, attacked_lp = self.encode_branches(attacked_x)
        clean_gates = self.gate(clean_cap, clean_dino, clean_lp)
        attacked_gates = self.gate(attacked_cap, attacked_dino, attacked_lp)
        # No branch dropout in the repair: first preserve the known-good v6 path.
        clean_fused = self.fuse_from_branches(clean_cap, clean_dino, clean_lp, clean_gates)
        attacked_fused = self.fuse_from_branches(
            attacked_cap, attacked_dino, attacked_lp, clean_gates.detach()
        )
        clean_logits = self.fusion(clean_fused)
        attacked_logits = self.fusion(attacked_fused)
        clean_soft = self.quantizer(clean_logits, hard=False)
        attacked_soft = self.quantizer(attacked_logits, hard=False)
        clean_hard = self.quantizer(clean_logits, hard=True)
        attacked_hard = self.quantizer(attacked_logits, hard=True)
        return {
            "clean_soft": clean_soft,
            "attacked_soft": attacked_soft,
            "clean_hard": clean_hard,
            "attacked_hard": attacked_hard,
            "clean_logits": clean_logits,
            "attacked_logits": attacked_logits,
            "clean_fused": clean_fused,
            "attacked_fused": attacked_fused,
            "clean_gates": clean_gates,
            "attacked_gates": attacked_gates,
            "clean_branches": (clean_cap, clean_dino, clean_lp),
            "attacked_branches": (attacked_cap, attacked_dino, attacked_lp),
        }


def _soft_bit_flip(clean: Tensor, attacked: Tensor) -> Tensor:
    # Bernoulli disagreement probability; smooth everywhere and aligned with
    # expected Hamming distance of sampled binary bits.
    c = clean.clamp(1e-4, 1 - 1e-4)
    a = attacked.clamp(1e-4, 1 - 1e-4)
    return c * (1 - a) + (1 - c) * a


def _contrastive_pair_loss(clean: Tensor, attacked: Tensor, temperature: float) -> Tensor:
    z = F.normalize(torch.cat([clean, attacked], dim=0), dim=-1)
    n = z.shape[0]
    logits = (z @ z.T) / max(float(temperature), 1e-3)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    positive = (torch.arange(n, device=z.device) + n // 2) % n
    logits = logits.masked_fill(eye, float("-inf"))
    return F.cross_entropy(logits, positive)


def _pairwise_separation(codes: Tensor, labels: Tensor, q05_target: float, q10_target: float):
    dist = _normalized_hamming(codes, codes)
    neg = dist.masked_fill(labels[:, None].eq(labels[None, :]), float("inf"))
    finite = neg[torch.isfinite(neg)]
    if finite.numel() == 0:
        z = codes.new_zeros(())
        return z, z, z, z
    nearest = neg.min(dim=1).values
    discr = F.softplus((0.32 - nearest) / 0.08).mul(0.08).mean()
    q05 = torch.quantile(finite, 0.05)
    q10 = torch.quantile(finite, 0.10)
    tail = (
        F.softplus((float(q05_target) - q05) / 0.025).mul(0.025)
        + 0.5 * F.softplus((float(q10_target) - q10) / 0.025).mul(0.025)
    )
    return discr, tail, q05, q10


def _binary_collision(codes: Tensor, labels: Tensor, target: float, topk: int) -> Tensor:
    dist = _normalized_hamming(codes, codes)
    neg = dist.masked_fill(labels[:, None].eq(labels[None, :]), float("inf"))
    fill = torch.where(torch.isfinite(neg), neg, torch.full_like(neg, 2.0))
    kk = min(max(int(topk), 1), fill.shape[1])
    vals = torch.topk(fill, kk, largest=False, dim=1).values
    tail = F.relu(float(target) - vals[:, 0]).pow(2)
    mass = F.relu(float(target) - vals).pow(2).mean(dim=1)
    return (0.70 * tail + 0.30 * mass).mean()


def _bit_quality(codes: Tensor, entropy_target: float, balance_target: float):
    hard = (codes >= 0.5).to(codes.dtype).detach() - codes.detach() + codes
    p = hard.mean(dim=0).clamp(1e-5, 1 - 1e-5)
    entropy = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p)).mean()
    balance = (p - 0.5).abs().mean()
    entropy_loss = F.relu(float(entropy_target) - entropy).pow(2)
    balance_loss = F.relu(balance - float(balance_target)).pow(2)
    centered = hard - hard.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-2)
    z = centered / std
    corr = (z.T @ z) / max(hard.shape[0], 1)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    decor = ((corr - eye) * (1 - eye)).pow(2).mean()
    return entropy_loss, balance_loss, decor, entropy.detach(), balance.detach()


def stable_objective(
    pair: dict[str, Tensor],
    labels: Tensor,
    memory_codes: Tensor | None,
    memory_labels: Tensor | None,
    config: StableConfig,
    stage: int = 2,
):
    clean = _finite(pair["clean_soft"], 1.0)
    attacked = _finite(pair["attacked_soft"], 1.0)
    clean_h = _finite(pair["clean_hard"], 1.0)
    attacked_h = _finite(pair["attacked_hard"], 1.0)
    fused_clean = _finite(pair["clean_fused"])
    fused_att = _finite(pair["attacked_fused"])

    flip = _soft_bit_flip(clean, attacked)
    robustness = flip.mean()
    robustness_q = torch.quantile(flip, 0.90)
    branch_consistency = torch.stack([
        _safe_cosine_loss(a, b) for a, b in zip(pair["clean_branches"], pair["attacked_branches"])
    ]).mean()
    contrastive = _contrastive_pair_loss(fused_clean, fused_att, config.contrastive_temperature)

    codes_s = torch.cat([clean, attacked], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    discr, tail, q05, q10 = _pairwise_separation(codes_s, labs, config.q05_target, config.q10_target)
    codes_h = torch.cat([clean_h, attacked_h], dim=0)
    binary_collision = _binary_collision(codes_h, labs, config.binary_collision_target, config.topk_negatives)
    entropy_loss, balance_loss, decor, entropy_obs, balance_obs = _bit_quality(
        codes_h, config.entropy_target, config.balance_target
    )
    quantization = ((clean - clean_h.detach()) ** 2 + (attacked - attacked_h.detach()) ** 2).mean()
    mean_gate = pair["clean_gates"].mean(dim=0)
    gate_usage = F.relu(0.15 - mean_gate).pow(2).mean()

    # Stage 1 protects the existing representation. Stage 2 gradually restores
    # collision pressure once robustness/representation alignment is established.
    collision_scale = 1.0 if stage >= 2 else 0.20
    loss = (
        config.lambda_robust * robustness
        + config.lambda_robust_q * F.relu(robustness_q - config.robust_q_target).pow(2)
        + config.lambda_contrastive * contrastive
        + config.lambda_branch_consistency * branch_consistency
        + collision_scale * (
            config.lambda_discrimination * discr
            + config.lambda_tail * tail
            + config.lambda_binary_collision * binary_collision
            + config.lambda_entropy * entropy_loss
            + config.lambda_balance * balance_loss
            + config.lambda_decorrelation * decor
        )
        + config.lambda_quantization * quantization
        + config.lambda_gate_usage * gate_usage
    )
    loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
    terms = {
        "robustness": robustness,
        "robustness_q": robustness_q,
        "contrastive": contrastive,
        "branch_consistency": branch_consistency,
        "discrimination": discr,
        "tail": tail,
        "binary_collision": binary_collision,
        "entropy": entropy_loss,
        "balance": balance_loss,
        "decorrelation": decor,
        "quantization": quantization,
        "gate_usage": gate_usage,
        "observed_entropy": entropy_obs,
        "observed_balance": balance_obs,
        "observed_q05": q05.detach(),
        "observed_q10": q10.detach(),
        "observed_gate_mean_cap": mean_gate[0].detach(),
        "observed_gate_mean_dino": mean_gate[1].detach(),
        "observed_gate_mean_logpolar": mean_gate[2].detach(),
    }
    return loss, terms


__all__ = ["STABLE_VERSION", "StableConfig", "CAPDinoLogPolarStable", "stable_objective"]
