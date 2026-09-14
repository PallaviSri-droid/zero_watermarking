from __future__ import annotations

"""CAP-DINO-LogPolar v7: collision-tail + hard-robustness + anti-dominance fusion.

Controlled refinement of v6. The critical v6 issue is that its hard robustness
quantile/worst-case tail was detached before optimization, so that tail statistic
was logged but could not provide a gradient. v7 keeps the same branches/backbone,
adds trainable hard-tail pressure, a residual cross-branch interaction path, and
explicit gate anti-dominance constraints.

No individual component is claimed as novel. The research hypothesis is the
collision-aware joint objective/evaluation and controlled fusion design.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cap_dino_logpolar_v6 import CAPDinoLogPolarV6, V6Config, _finite, _safe_cosine_loss
from .training import _normalized_hamming

CAP_DINO_LP_V7_VERSION = "CAP-DINO-LP-v7"


@dataclass
class V7Config(V6Config):
    quantizer_temperature: float = 1.35
    threshold_init_std: float = 0.10
    dropout: float = 0.08
    lr: float = 6e-5
    attack_views: int = 11

    # Keep v6 robustness but make tail risk trainable.
    robust_target: float = 0.034
    robust_q_target: float = 0.060
    robust_hard_target: float = 0.055
    robust_hard_q: float = 0.90
    robust_hard_tail_fraction: float = 0.20
    lambda_robust: float = 0.90
    lambda_robust_q: float = 0.70
    lambda_robust_hard: float = 1.10
    lambda_robust_tail: float = 1.00

    # Controls carried forward from the stronger v12-v14 candidates.
    q05_target: float = 0.10
    q10_target: float = 0.16
    entropy_target: float = 0.86
    balance_target: float = 0.105
    binary_collision_target: float = 0.18
    topk_negatives: int = 32
    memory_size: int = 8192
    lambda_discrimination: float = 1.00
    lambda_tail: float = 1.15
    lambda_binary_collision: float = 1.30
    lambda_entropy: float = 0.80
    lambda_balance: float = 0.45
    lambda_decorrelation: float = 0.35

    # Stop the CAP branch from becoming the only useful path.
    branch_dropout: float = 0.25
    gate_min_usage: float = 0.18
    gate_max_usage: float = 0.48
    gate_std_target: float = 0.055
    gate_row_max_target: float = 0.58
    lambda_gate_usage: float = 0.30
    lambda_gate_diversity: float = 0.20
    lambda_gate_dominance: float = 0.25

    # Complement the convex mixture with controlled branch interactions.
    mixer_dropout: float = 0.08
    mixer_scale_init: float = -1.20
    lambda_mixer_consistency: float = 0.35


class CAPDinoLogPolarV7(CAPDinoLogPolarV6):
    """v7 with residual branch interaction and stronger collision/robustness tails."""

    def __init__(self, config: V7Config | None = None) -> None:
        super().__init__(config or V7Config())
        c = self.config
        self.branch_mixer = nn.Sequential(
            nn.LayerNorm(c.fusion_dim * 3),
            nn.Linear(c.fusion_dim * 3, c.fusion_dim),
            nn.GELU(),
            nn.Dropout(c.mixer_dropout),
            nn.Linear(c.fusion_dim, c.fusion_dim),
            nn.GELU(),
        ).to(self.runtime_device)
        self.mixer_scale = nn.Parameter(torch.tensor(float(c.mixer_scale_init)))

    @staticmethod
    def _gated_concat(cap: Tensor, dino: Tensor, lp: Tensor, gates: Tensor) -> Tensor:
        return torch.cat(
            [gates[:, 0:1] * cap, gates[:, 1:2] * dino, gates[:, 2:3] * lp],
            dim=-1,
        )

    def _fused_v7(self, cap: Tensor, dino: Tensor, lp: Tensor, gates: Tensor) -> tuple[Tensor, Tensor]:
        weighted = self._gated_concat(cap, dino, lp, gates)
        d = cap.shape[1]
        convex = weighted[:, :d] + weighted[:, d:2 * d] + weighted[:, 2 * d:]
        interaction = self.branch_mixer(weighted)
        scale = torch.sigmoid(self.mixer_scale)
        return convex + scale * interaction, interaction

    def forward_pair(self, clean_x: Tensor, attacked_x: Tensor):
        clean_cap, clean_dino, clean_lp = self.encode_branches(clean_x)
        attacked_cap, attacked_dino, attacked_lp = self.encode_branches(attacked_x)
        clean_gates = self.gate(clean_cap, clean_dino, clean_lp)
        attacked_gates = self.gate(attacked_cap, attacked_dino, attacked_lp)
        shared_gates, branch_mask = self._shared_branch_dropout(clean_gates)
        clean_fused, clean_interaction = self._fused_v7(clean_cap, clean_dino, clean_lp, shared_gates)
        attacked_fused, attacked_interaction = self._fused_v7(attacked_cap, attacked_dino, attacked_lp, shared_gates.detach())
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
            "clean_interaction": clean_interaction,
            "attacked_interaction": attacked_interaction,
        }

    def forward(self, x: Tensor, hard: bool = False) -> Tensor:
        cap, dino, lp = self.encode_branches(x)
        gates = self.gate(cap, dino, lp)
        fused, _ = self._fused_v7(cap, dino, lp, gates)
        return self.quantizer(self.fusion(fused), hard=hard)


def _hard_robustness_terms(clean: Tensor, attacked: Tensor, config: V7Config) -> tuple[Tensor, Tensor, Tensor]:
    """Trainable hard-code mean/q90/worst robustness terms."""
    d = _finite((clean - attacked).abs().mean(dim=1), 2.0)
    beta = max(float(config.robust_softness), 1e-3)
    target = float(config.robust_hard_target)
    mean_loss = F.softplus((d.mean() - target) / beta).mul(beta).clamp_max(2.0)
    q = torch.quantile(d, float(min(max(config.robust_hard_q, 0.5), 0.99)))
    q_loss = F.softplus((q - target) / beta).mul(beta).clamp_max(2.0)
    k = max(1, min(d.numel(), int(round(d.numel() * config.robust_hard_tail_fraction))))
    worst = torch.topk(d, k, largest=True).values.mean()
    worst_loss = F.softplus((worst - target) / beta).mul(beta).clamp_max(2.0)
    return mean_loss, q_loss + 0.50 * worst_loss, d


def _ste_hard(codes: Tensor) -> Tensor:
    return (codes >= 0.5).to(codes.dtype).detach() - codes.detach() + codes


def _hard_bit_constraints(codes: Tensor, entropy_target: float, balance_target: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hard = _ste_hard(codes)
    p = hard.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    entropy_loss = F.relu(float(entropy_target) - entropy).pow(2)
    balance_loss = F.relu(balance - float(balance_target)).pow(2)
    return entropy_loss, balance_loss, entropy.detach(), balance.detach()


def _pairwise_tail_loss(codes: Tensor, labels: Tensor, q05_target: float, q10_target: float) -> tuple[Tensor, Tensor, Tensor]:
    dist = _normalized_hamming(codes, codes)
    neg = dist.masked_fill(labels[:, None].eq(labels[None, :]), float("inf"))
    finite = neg[torch.isfinite(neg)]
    if finite.numel() == 0:
        z = codes.new_zeros(())
        return z, z, z
    q05 = torch.quantile(finite, 0.05)
    q10 = torch.quantile(finite, 0.10)
    tail = (
        F.softplus((float(q05_target) - q05) / 0.025).mul(0.025)
        + 0.50 * F.softplus((float(q10_target) - q10) / 0.025).mul(0.025)
    ).clamp_max(2.0)
    nearest = neg.min(dim=1).values
    discrimination = F.softplus((0.36 - nearest) / 0.08).mul(0.08).mean().clamp_max(2.0)
    return tail, discrimination, finite


def _binary_collision(codes: Tensor, labels: Tensor, target: float, topk: int) -> Tensor:
    dist = _normalized_hamming(codes, codes)
    neg = dist.masked_fill(labels[:, None].eq(labels[None, :]), float("inf"))
    fill = torch.where(torch.isfinite(neg), neg, torch.full_like(neg, 2.0))
    kk = min(max(int(topk), 1), fill.shape[1])
    vals = torch.topk(fill, kk, largest=False, dim=1).values
    tail = F.relu(float(target) - vals[:, 0]).pow(2)
    mass = F.relu(float(target) - vals).pow(2).mean(dim=1)
    return (0.70 * tail + 0.30 * mass).mean().clamp_max(1.0)


def _memory_binary_collision(codes: Tensor, labels: Tensor, memory_codes: Tensor | None, memory_labels: Tensor | None, target: float, topk: int) -> Tensor:
    if memory_codes is None or memory_codes.numel() == 0:
        return codes.new_zeros(())
    md = _normalized_hamming(codes, memory_codes.detach().clamp(0.0, 1.0))
    if memory_labels is not None:
        md = md.masked_fill(labels[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
    fill = torch.where(torch.isfinite(md), md, torch.full_like(md, 2.0))
    kk = min(max(int(topk), 1), fill.shape[1])
    vals = torch.topk(fill, kk, largest=False, dim=1).values
    tail = F.relu(float(target) - vals[:, 0]).pow(2)
    mass = F.relu(float(target) - vals).pow(2).mean(dim=1)
    return (0.70 * tail + 0.30 * mass).mean().clamp_max(1.0)


def v7_objective(pair: dict[str, Tensor], labels: Tensor, memory_codes: Tensor | None, memory_labels: Tensor | None, config: V7Config) -> tuple[Tensor, dict[str, Tensor]]:
    clean_s = _finite(pair["clean_soft"], 1.0)
    attacked_s = _finite(pair["attacked_soft"], 1.0)
    clean_h = _finite(pair["clean_hard"], 1.0)
    attacked_h = _finite(pair["attacked_hard"], 1.0)
    pair_s = torch.cat([clean_s, attacked_s], dim=0)
    pair_h = torch.cat([clean_h, attacked_h], dim=0)
    pair_labels = torch.cat([labels, labels], dim=0)

    soft_d = (clean_s - attacked_s).abs().mean(dim=1)
    soft_rob = soft_d.mean()
    soft_q = torch.quantile(soft_d, 0.90)
    soft_q_loss = F.softplus((soft_q - config.robust_q_target) / config.robust_softness).mul(config.robust_softness).clamp_max(2.0)
    hard_rob, hard_tail, hard_d = _hard_robustness_terms(clean_h, attacked_h, config)

    branch_consistency = torch.stack([
        _safe_cosine_loss(a, b) for a, b in zip(pair["clean_branches"], pair["attacked_branches"])
    ]).mean()
    gate_consistency = F.mse_loss(pair["clean_gates"], pair["attacked_gates"]).clamp_max(2.0)
    tail, discrimination, inter = _pairwise_tail_loss(pair_s, pair_labels, config.q05_target, config.q10_target)

    hard_collision = _binary_collision(pair_h, pair_labels, config.binary_collision_target, config.topk_negatives)
    memory_collision = _memory_binary_collision(pair_h, pair_labels, memory_codes, memory_labels, config.binary_collision_target, config.topk_negatives)

    hard_entropy, hard_balance, entropy_obs, balance_obs = _hard_bit_constraints(pair_h, config.entropy_target, config.balance_target)
    p = pair_h.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    soft_entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    soft_balance = (p - 0.5).abs().mean()
    centered = pair_h - pair_h.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-2)
    z = centered / std
    corr = (z.T @ z) / max(pair_h.shape[0], 1)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    decor = ((corr - eye) * (1.0 - eye)).pow(2).mean().clamp_max(2.0)

    gates = pair["clean_gates"]
    mean_g = gates.mean(dim=0)
    gate_usage = (F.relu(config.gate_min_usage - mean_g).pow(2) + F.relu(mean_g - config.gate_max_usage).pow(2)).mean()
    gate_std = gates.std(dim=0, unbiased=False).mean()
    gate_diversity = F.relu(config.gate_std_target - gate_std).pow(2)
    gate_dominance = F.relu(gates.max(dim=1).values - config.gate_row_max_target).pow(2).mean()

    clean_i = pair["clean_interaction"]
    attacked_i = pair["attacked_interaction"]
    mixer_consistency = _safe_cosine_loss(clean_i, attacked_i)

    loss = (
        config.lambda_robust * soft_rob
        + config.lambda_robust_q * soft_q_loss
        + config.lambda_robust_hard * hard_rob
        + config.lambda_robust_tail * hard_tail
        + config.lambda_branch_consistency * branch_consistency
        + config.lambda_gate_consistency * (gate_consistency + gate_dominance)
        + config.lambda_attack_consistency * gate_consistency
        + config.lambda_discrimination * discrimination
        + config.lambda_tail * tail
        + config.lambda_binary_collision * (0.70 * hard_collision + 0.30 * memory_collision)
        + config.lambda_entropy * (0.5 * hard_entropy + 0.5 * F.relu(config.entropy_target - soft_entropy).pow(2))
        + config.lambda_balance * (0.5 * hard_balance + 0.5 * F.relu(soft_balance - config.balance_target).pow(2))
        + config.lambda_decorrelation * decor
        + config.lambda_gate_usage * gate_usage
        + config.lambda_gate_diversity * gate_diversity
        + config.lambda_gate_dominance * gate_dominance
        + config.lambda_mixer_consistency * mixer_consistency
    )
    loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    terms = {
        "robustness": soft_rob,
        "robustness_q": soft_q_loss,
        "robustness_hard": hard_rob,
        "robustness_hard_tail": hard_tail,
        "branch_consistency": branch_consistency,
        "gate_consistency": gate_consistency,
        "discrimination": discrimination,
        "tail": tail,
        "binary_collision": 0.70 * hard_collision + 0.30 * memory_collision,
        "entropy": 0.5 * hard_entropy + 0.5 * F.relu(config.entropy_target - soft_entropy).pow(2),
        "balance": 0.5 * hard_balance + 0.5 * F.relu(soft_balance - config.balance_target).pow(2),
        "decorrelation": decor,
        "gate_usage": gate_usage,
        "gate_diversity": gate_diversity,
        "gate_dominance": gate_dominance,
        "mixer_consistency": mixer_consistency,
        "observed_entropy": entropy_obs,
        "observed_balance": balance_obs,
        "observed_robustness": soft_d.mean().detach(),
        "observed_hard_robustness": hard_d.detach().mean(),
        "observed_gate_std": gate_std.detach(),
        "observed_gate_max": gates.max(dim=1).values.detach().mean(),
        "observed_q05": torch.quantile(inter.detach(), 0.05) if inter.numel() else clean_s.new_tensor(0.0),
        "observed_q10": torch.quantile(inter.detach(), 0.10) if inter.numel() else clean_s.new_tensor(0.0),
    }
    return loss, terms


__all__ = ["CAP_DINO_LP_V7_VERSION", "V7Config", "CAPDinoLogPolarV7", "v7_objective"]
