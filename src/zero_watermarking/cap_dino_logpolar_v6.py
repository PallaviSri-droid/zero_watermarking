from __future__ import annotations

"""CAP-DINO-LogPolar v6: tail-robust, branch-regularized adaptive fusion.

Numerical-stability patch: all normalization-sensitive terms are finite-guarded,
robustness constraints are computed from detached hard values for metrics while
retaining smooth soft gradients, and optimizer-facing loss terms are clamped.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .cap_dino_logpolar_v5 import CAPDinoLogPolarV5, V5Config
from .training import _normalized_hamming

CAP_DINO_LP_V6_VERSION = "CAP-DINO-LP-v6"


@dataclass
class V6Config(V5Config):
    attack_views: int = 11
    robust_target: float = 0.035
    robust_q_target: float = 0.065
    robust_hard_target: float = 0.060
    robust_hard_q: float = 0.90
    robust_hard_tail_fraction: float = 0.15
    lambda_robust_hard: float = 0.75
    lambda_robust_tail: float = 0.60
    branch_dropout: float = 0.15
    lambda_gate_diversity: float = 0.12
    gate_min_usage: float = 0.15
    gate_max_usage: float = 0.65
    gate_std_target: float = 0.035
    lambda_attack_consistency: float = 0.30
    lambda_input_consistency: float = 0.20
    temperature: float = 0.07


class CAPDinoLogPolarV6(CAPDinoLogPolarV5):
    """v6 with stable tail robustness and branch regularization."""

    def __init__(self, config: V6Config | None = None) -> None:
        super().__init__(config or V6Config())

    def _shared_branch_dropout(self, gates: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training or self.config.branch_dropout <= 0:
            return gates, torch.ones_like(gates)
        p = float(min(max(self.config.branch_dropout, 0.0), 0.45))
        keep = (torch.rand_like(gates) > p).to(gates.dtype)
        for row in range(keep.shape[0]):
            if int(keep[row].sum().item()) < 2:
                order = torch.argsort(torch.rand(3, device=gates.device), descending=True)
                keep[row, order[:2]] = 1.0
        dropped = gates * keep
        dropped = dropped / dropped.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return dropped, keep

    def forward_pair(self, clean_x: Tensor, attacked_x: Tensor):
        clean_cap, clean_dino, clean_lp = self.encode_branches(clean_x)
        attacked_cap, attacked_dino, attacked_lp = self.encode_branches(attacked_x)
        clean_gates = self.gate(clean_cap, clean_dino, clean_lp)
        attacked_gates = self.gate(attacked_cap, attacked_dino, attacked_lp)
        shared_gates, branch_mask = self._shared_branch_dropout(clean_gates)
        clean_fused = self.fuse_from_branches(clean_cap, clean_dino, clean_lp, shared_gates)
        attacked_fused = self.fuse_from_branches(attacked_cap, attacked_dino, attacked_lp, shared_gates.detach())
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
            "clean_gates": clean_gates,
            "attacked_gates": attacked_gates,
            "shared_gates": shared_gates,
            "branch_mask": branch_mask,
            "clean_branches": (clean_cap, clean_dino, clean_lp),
            "attacked_branches": (attacked_cap, attacked_dino, attacked_lp),
        }


def _finite(x: Tensor, limit: float = 1e3) -> Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def _hard_robustness_tail(clean: Tensor, attacked: Tensor, config: V6Config) -> tuple[Tensor, Tensor, Tensor]:
    d = _finite((clean - attacked).abs().mean(dim=1), 10.0)
    mean_d = d.mean()
    q = torch.quantile(d.detach(), float(min(max(config.robust_hard_q, 0.5), 0.99)))
    k = max(1, int(round(float(d.numel()) * config.robust_hard_tail_fraction)))
    worst = torch.topk(d.detach(), min(k, d.numel()), largest=True).values.mean()
    beta = max(float(config.robust_softness), 1e-3)
    mean_loss = F.softplus((mean_d - config.robust_hard_target) / beta).mul(beta).clamp_max(2.0)
    q_loss = F.softplus((q - config.robust_hard_target) / beta).mul(beta).clamp_max(2.0)
    worst_loss = F.softplus((worst - config.robust_hard_target) / beta).mul(beta).clamp_max(2.0)
    return mean_loss, q_loss + 0.50 * worst_loss, d.detach()


def _safe_cosine_loss(a: Tensor, b: Tensor) -> Tensor:
    a = _finite(a)
    b = _finite(b)
    na = a.norm(dim=1).clamp_min(1e-6)
    nb = b.norm(dim=1).clamp_min(1e-6)
    cos = (a * b).sum(dim=1) / (na * nb)
    return (1.0 - cos.clamp(-1.0, 1.0)).mean()


def v6_objective(pair: dict[str, Tensor], labels: Tensor, memory_codes: Tensor | None,
                 memory_labels: Tensor | None, config: V6Config) -> tuple[Tensor, dict[str, Tensor]]:
    clean_s = _finite(pair["clean_soft"], 1.0)
    attacked_s = _finite(pair["attacked_soft"], 1.0)
    clean_h = _finite(pair["clean_hard"], 1.0)
    attacked_h = _finite(pair["attacked_hard"], 1.0)
    clean_g = _finite(pair["clean_gates"], 1.0)
    attacked_g = _finite(pair["attacked_gates"], 1.0)

    soft_d = (clean_s - attacked_s).abs().mean(dim=1)
    soft_mean = soft_d.mean().clamp_max(2.0)
    soft_q = torch.quantile(soft_d.detach(), 0.90)
    hard_mean, hard_tail, hard_d = _hard_robustness_tail(clean_h, attacked_h, config)

    clean_br = tuple(_finite(x) for x in pair["clean_branches"])
    attacked_br = tuple(_finite(x) for x in pair["attacked_branches"])
    branch_cos = torch.stack([_safe_cosine_loss(a, b) for a, b in zip(clean_br, attacked_br)]).mean().clamp_max(2.0)
    gate_consistency = F.mse_loss(clean_g, attacked_g).clamp_max(2.0)

    codes_soft = torch.cat([clean_s, attacked_s], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    dist = _normalized_hamming(codes_soft, codes_soft)
    neg = dist.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    finite = neg[torch.isfinite(neg)]
    if finite.numel():
        nearest = neg.min(dim=1).values
        discrimination = F.softplus((config.margin - nearest) / 0.08).mul(0.08).mean().clamp_max(2.0)
        q05 = torch.quantile(finite.detach(), 0.05)
        q10 = torch.quantile(finite.detach(), 0.10)
        tail = (
            F.softplus((config.q05_target - q05) / 0.025).mul(0.025)
            + 0.5 * F.softplus((config.q10_target - q10) / 0.025).mul(0.025)
        ).clamp_max(2.0)
    else:
        discrimination = clean_s.new_zeros(())
        tail = clean_s.new_zeros(())

    codes_h = torch.cat([clean_h, attacked_h], dim=0)
    h_labels = torch.cat([labels, labels], dim=0)
    hdist = _normalized_hamming(codes_h, codes_h)
    hneg = hdist.masked_fill(h_labels[:, None].eq(h_labels[None, :]), float("inf"))
    if memory_codes is not None and memory_codes.numel():
        mem = _finite(memory_codes.detach(), 1.0)
        memd = _normalized_hamming(codes_h, mem)
        if memory_labels is not None:
            memd = memd.masked_fill(h_labels[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
        hneg = torch.cat([hneg, memd], dim=1)
    fill = torch.where(torch.isfinite(hneg), hneg, torch.full_like(hneg, 2.0))
    kk = min(max(int(config.topk_negatives), 1), fill.shape[1])
    top = torch.topk(fill, kk, largest=False, dim=1).values
    binary_tail = F.relu(config.binary_collision_target - top[:, 0]).pow(2).mean().clamp_max(1.0)
    binary_mass = F.relu(config.binary_collision_target - top).pow(2).mean().clamp_max(1.0)

    p = codes_h.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    centered = codes_h - codes_h.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-2)
    z = centered / std
    corr = (z.T @ z) / max(codes_h.shape[0], 1)
    corr = _finite(corr, 10.0)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    decor = ((corr - eye) * (1.0 - eye)).pow(2).mean().clamp_max(2.0)
    entropy_loss = F.relu(config.entropy_target - entropy).pow(2).clamp_max(2.0)
    balance_loss = F.relu(balance - config.balance_target).pow(2).clamp_max(2.0)

    mean_gate = clean_g.mean(dim=0)
    gate_std = clean_g.std(dim=0, unbiased=False).mean()
    gate_usage = (
        F.relu(config.gate_min_usage - mean_gate).pow(2).mean()
        + F.relu(mean_gate - config.gate_max_usage).pow(2).mean()
    ).clamp_max(1.0)
    gate_diversity = F.relu(config.gate_std_target - gate_std).pow(2).clamp_max(1.0)
    attacked_shift = torch.stack([_safe_cosine_loss(a, b) for a, b in zip(clean_br, attacked_br)]).mean().clamp_max(2.0)

    terms = {
        "robustness": soft_mean,
        "robustness_q": soft_q,
        "robustness_hard": hard_mean.detach(),
        "robustness_hard_tail": hard_tail.detach(),
        "branch_consistency": branch_cos,
        "gate_consistency": gate_consistency,
        "discrimination": discrimination,
        "tail": tail,
        "binary_collision": (0.70 * binary_tail + 0.30 * binary_mass).clamp_max(1.0),
        "entropy": entropy_loss,
        "balance": balance_loss,
        "decorrelation": decor,
        "gate_usage": gate_usage,
        "gate_diversity": gate_diversity,
        "attack_consistency": attacked_shift,
        "observed_entropy": entropy.detach(),
        "observed_balance": balance.detach(),
        "observed_robustness": soft_d.mean().detach(),
        "observed_hard_robustness": hard_d.mean(),
        "observed_gate_std": gate_std.detach(),
    }
    soft_q_loss = F.softplus((soft_q - config.robust_q_target) / config.robust_softness).mul(config.robust_softness).clamp_max(2.0)
    loss = (
        config.lambda_robust * soft_mean
        + config.lambda_robust_q * soft_q_loss
        + config.lambda_robust_hard * hard_mean
        + config.lambda_robust_tail * hard_tail
        + config.lambda_branch_consistency * branch_cos
        + config.lambda_gate_consistency * gate_consistency
        + config.lambda_attack_consistency * attacked_shift
        + config.lambda_discrimination * discrimination
        + config.lambda_tail * tail
        + config.lambda_binary_collision * (0.70 * binary_tail + 0.30 * binary_mass)
        + config.lambda_entropy * entropy_loss
        + config.lambda_balance * balance_loss
        + config.lambda_decorrelation * decor
        + config.lambda_gate_usage * gate_usage
        + config.lambda_gate_diversity * gate_diversity
    )
    loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
    return loss, terms


__all__ = ["CAP_DINO_LP_V6_VERSION", "V6Config", "CAPDinoLogPolarV6", "v6_objective"]
