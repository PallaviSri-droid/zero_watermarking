from __future__ import annotations

"""CAP-DINO-LogPolar v5: robustness-preserving adaptive fusion.

v5 addresses measured v4 failure modes rather than changing components blindly:
- shared clean-image gates are reused for attacked views, stabilizing routing;
- binary STE codes are optimized for collision metrics seen at evaluation;
- memory-bank negatives increase cross-image coverage;
- explicit lower-tail separation, hard entropy/balance, and robustness-quantile
  constraints combine the strongest ideas observed in CAP v12-v14;
- branch-level attack consistency prevents one branch from becoming brittle;
- the gate is no longer rewarded for uniform entropy; a weak usage floor avoids
  branch death while allowing genuine image-adaptive specialization.

DINOv2, Log-Polar and CAP are established components. The research candidate is
this collision-aware, attack-consistent multi-objective fusion and evaluation.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cap_dino_logpolar import CAPDinoLogPolar, FusionConfig
from .training import _normalized_hamming

CAP_DINO_LP_V5_VERSION = "CAP-DINO-LP-v5"


@dataclass
class V5Config(FusionConfig):
    quantizer_temperature: float = 1.5
    threshold_init_std: float = 0.08
    dropout: float = 0.08
    robust_target: float = 0.055
    robust_q_target: float = 0.075
    robust_softness: float = 0.012
    margin: float = 0.36
    q05_target: float = 0.10
    q10_target: float = 0.16
    entropy_target: float = 0.82
    balance_target: float = 0.12
    binary_confidence_target: float = 0.16
    binary_collision_target: float = 0.16
    memory_size: int = 4096
    topk_negatives: int = 24
    attack_views: int = 6
    lambda_robust: float = 1.00
    lambda_robust_q: float = 0.65
    lambda_branch_consistency: float = 0.60
    lambda_discrimination: float = 1.00
    lambda_tail: float = 1.10
    lambda_binary_collision: float = 1.15
    lambda_entropy: float = 0.70
    lambda_balance: float = 0.30
    lambda_decorrelation: float = 0.25
    lambda_gate_consistency: float = 0.30
    lambda_gate_usage: float = 0.20
    lambda_gate_sharpness: float = 0.04


class CAPDinoLogPolarV5(CAPDinoLogPolar):
    """v5 model with richer trainable medical-domain adapters."""

    def __init__(self, config: V5Config | None = None) -> None:
        super().__init__(config or V5Config())
        c = self.config
        hidden = max(96, c.fusion_dim // 2)
        self.dino_project = nn.Sequential(
            nn.LayerNorm(c.dino_dim),
            nn.Linear(c.dino_dim, hidden), nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(hidden, c.fusion_dim), nn.GELU(),
        ).to(self.runtime_device)
        self.logpolar_project = nn.Sequential(
            nn.LayerNorm(32 * 32),
            nn.Linear(32 * 32, hidden), nn.GELU(),
            nn.Linear(hidden, c.fusion_dim), nn.GELU(),
        ).to(self.runtime_device)
        # CAP already has a compact trainable encoder; give its projection a
        # nonlinear adapter so fusion does not force all domain adaptation into
        # the final hash head.
        self.cap_project = nn.Sequential(
            nn.LayerNorm(c.cap_dim),
            nn.Linear(c.cap_dim, hidden), nn.GELU(),
            nn.Linear(hidden, c.fusion_dim), nn.GELU(),
        ).to(self.runtime_device)

    def encode_branches(self, x: Tensor):
        cap = F.normalize(self.cap_project(self.cap(x, hard=False)), dim=-1) * self.config.cap_weight
        dino = F.normalize(self.dino_project(self.dino_features(x)), dim=-1) * self.config.dino_weight
        logpolar = F.normalize(self.logpolar_project(self.logpolar_features(x)), dim=-1) * self.config.logpolar_weight
        return cap, dino, logpolar

    def fuse_from_branches(self, cap: Tensor, dino: Tensor, logpolar: Tensor, gates: Tensor) -> Tensor:
        return gates[:, 0:1] * cap + gates[:, 1:2] * dino + gates[:, 2:3] * logpolar

    def forward_pair(self, clean_x: Tensor, attacked_x: Tensor):
        clean_cap, clean_dino, clean_lp = self.encode_branches(clean_x)
        attacked_cap, attacked_dino, attacked_lp = self.encode_branches(attacked_x)
        clean_gates = self.gate(clean_cap, clean_dino, clean_lp)
        attacked_gates = self.gate(attacked_cap, attacked_dino, attacked_lp)
        # Critical v5 change: use the clean-image routing decision for the
        # attacked view. The representation should remain the same even when
        # the observed pixels are perturbed.
        clean_fused = self.fuse_from_branches(clean_cap, clean_dino, clean_lp, clean_gates)
        attacked_fused = self.fuse_from_branches(attacked_cap, attacked_dino, attacked_lp, clean_gates.detach())
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
            "clean_branches": (clean_cap, clean_dino, clean_lp),
            "attacked_branches": (attacked_cap, attacked_dino, attacked_lp),
        }

    def forward(self, x: Tensor, hard: bool = False) -> Tensor:
        cap, dino, lp = self.encode_branches(x)
        gates = self.gate(cap, dino, lp)
        fused = self.fuse_from_branches(cap, dino, lp, gates)
        logits = self.fusion(fused)
        return self.quantizer(logits, hard=hard)


def _ste_hard(code: Tensor) -> Tensor:
    hard = (code >= 0.5).to(code.dtype)
    return hard.detach() - code.detach() + code


def _entropy_balance(codes: Tensor, entropy_target: float, balance_target: float):
    hard = _ste_hard(codes)
    p = hard.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    entropy_loss = F.relu(float(entropy_target) - entropy).pow(2)
    balance_loss = F.relu(balance - float(balance_target)).pow(2)
    return entropy_loss, balance_loss, entropy.detach(), balance.detach()


def _decorrelation(codes: Tensor) -> Tensor:
    centered = codes - codes.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-3)
    z = centered / std
    corr = (z.T @ z) / max(codes.shape[0], 1)
    eye = torch.eye(corr.shape[0], device=codes.device, dtype=codes.dtype)
    return ((corr - eye) * (1.0 - eye)).pow(2).mean()


def _memory_collision(clean: Tensor, attacked: Tensor, labels: Tensor,
                      memory_codes: Tensor | None, memory_labels: Tensor | None,
                      target: float, topk: int):
    codes = torch.cat([_ste_hard(clean), _ste_hard(attacked)], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    dist = _normalized_hamming(codes, codes)
    dist = dist.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    sources = [dist]
    if memory_codes is not None and memory_codes.numel():
        mem = _ste_hard(memory_codes.detach())
        md = _normalized_hamming(codes, mem)
        if memory_labels is not None:
            md = md.masked_fill(labs[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
        sources.append(md)
    tail_terms = []
    mass_terms = []
    for src in sources:
        finite = src[torch.isfinite(src)]
        if finite.numel() == 0:
            continue
        rows = src.shape[0]
        kk = min(max(int(topk), 1), src.shape[1])
        fill = torch.where(torch.isfinite(src), src, torch.full_like(src, 2.0))
        vals = torch.topk(fill, kk, largest=False, dim=1).values
        tail_terms.append(F.relu(float(target) - vals[:, 0]).pow(2).mean())
        mass_terms.append(F.relu(float(target) - vals).pow(2).mean())
    zero = clean.new_zeros(())
    return (torch.stack(tail_terms).mean() if tail_terms else zero,
            torch.stack(mass_terms).mean() if mass_terms else zero)


def _robustness_terms(clean: Tensor, attacked: Tensor, config: V5Config):
    d = (clean - attacked).abs().mean(dim=1)
    mean = d.mean()
    q90 = torch.quantile(d, 0.90)
    worst = d.max()
    target = clean.new_tensor(config.robust_target)
    q_target = clean.new_tensor(config.robust_q_target)
    mean_loss = F.softplus((mean - target) / config.robust_softness).mul(config.robust_softness)
    q_loss = F.softplus((q90 - q_target) / config.robust_softness).mul(config.robust_softness)
    worst_loss = F.relu(worst - 2.0 * q_target).pow(2)
    return mean_loss, q_loss + 0.50 * worst_loss, d.detach()


def v5_objective(pair: dict[str, Tensor], labels: Tensor,
                 memory_codes: Tensor | None, memory_labels: Tensor | None,
                 config: V5Config) -> tuple[Tensor, dict[str, Tensor]]:
    clean_s, attacked_s = pair["clean_soft"], pair["attacked_soft"]
    clean_h, attacked_h = pair["clean_hard"], pair["attacked_hard"]
    mean_rob, tail_rob, robust_dist = _robustness_terms(clean_s, attacked_s, config)
    clean_e, att_e, clean_r, att_r = pair["clean_branches"], pair["attacked_branches"], pair["clean_gates"], pair["attacked_gates"]
    branch_consistency = sum(1.0 - (a * b).sum(dim=1).mean() for a, b in zip(clean_e, att_e)) / 3.0
    gate_consistency = F.mse_loss(clean_r, att_r)

    # Continuous separation complements the binary collision objective.
    codes_soft = torch.cat([clean_s, attacked_s], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    dist = _normalized_hamming(codes_soft, codes_soft)
    neg = dist.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    finite = neg[torch.isfinite(neg)]
    if finite.numel():
        nearest = neg.min(dim=1).values
        discrimination = F.softplus((config.margin - nearest) / 0.08).mul(0.08).mean()
        q05 = torch.quantile(finite, 0.05)
        q10 = torch.quantile(finite, 0.10)
        tail = F.softplus((config.q05_target - q05) / 0.025).mul(0.025) + 0.5 * F.softplus((config.q10_target - q10) / 0.025).mul(0.025)
    else:
        discrimination = clean_s.new_zeros(())
        tail = clean_s.new_zeros(())
    binary_tail, binary_mass = _memory_collision(clean_h, attacked_h, labels, memory_codes, memory_labels, config.binary_collision_target, config.topk_negatives)
    entropy_loss, balance_loss, entropy_obs, balance_obs = _entropy_balance(torch.cat([clean_h, attacked_h], dim=0), config.entropy_target, config.balance_target)
    decor = _decorrelation(torch.cat([clean_h, attacked_h], dim=0))
    mean_gate = clean_r.mean(dim=0)
    gate_usage = F.relu(0.20 - mean_gate).pow(2).mean()
    gate_entropy = -(clean_r.clamp_min(1e-8) * clean_r.clamp_min(1e-8).log()).sum(dim=1).mean()
    # Only a weak late-stage pressure away from exact uniform routing; unlike v4
    # this is not a reward for maximum entropy.
    gate_sharp = F.relu(gate_entropy - 1.05).pow(2)

    terms = {
        "robustness": mean_rob,
        "robustness_q": tail_rob,
        "branch_consistency": branch_consistency,
        "discrimination": discrimination,
        "tail": tail,
        "binary_collision": 0.70 * binary_tail + 0.30 * binary_mass,
        "entropy": entropy_loss,
        "balance": balance_loss,
        "decorrelation": decor,
        "gate_consistency": gate_consistency,
        "gate_usage": gate_usage,
        "gate_sharpness": gate_sharp,
        "observed_entropy": entropy_obs,
        "observed_balance": balance_obs,
        "observed_robustness": robust_dist.mean(),
    }
    loss = (
        config.lambda_robust * mean_rob
        + config.lambda_robust_q * tail_rob
        + config.lambda_branch_consistency * branch_consistency
        + config.lambda_discrimination * discrimination
        + config.lambda_tail * tail
        + config.lambda_binary_collision * (0.70 * binary_tail + 0.30 * binary_mass)
        + config.lambda_entropy * entropy_loss
        + config.lambda_balance * balance_loss
        + config.lambda_decorrelation * decor
        + config.lambda_gate_consistency * gate_consistency
        + config.lambda_gate_usage * gate_usage
        + config.lambda_gate_sharpness * gate_sharp
    )
    return loss, terms


__all__ = ["CAP_DINO_LP_V5_VERSION", "V5Config", "CAPDinoLogPolarV5", "v5_objective"]
