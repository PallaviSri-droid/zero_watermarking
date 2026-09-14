from __future__ import annotations

"""CAP-ZW-v12 research candidate.

This version keeps the v11 architecture but adds three explicit controls motivated
by the observed 200-image collapse and recent deep-hashing / zero-watermarking
literature:

1. lower-tail separation loss on continuous codes (not only the nearest negative),
2. explicit bit-entropy floor to prevent code collapse,
3. binary-confidence regularization to reduce ambiguous bits near 0.5.

The selective robustness guard is deliberately softened rather than removed.
This is a research candidate, not a claimed final model; selection remains
protocol-locked and multi-seed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .training import CAPZWHashNet, PairAttackDataset, TrainConfig, _update_memory, mgda_weights, objective_terms

CAP_ZW_V12_VERSION = "CAP-ZW-v12"


@dataclass
class V12Config(TrainConfig):
    """Collision-first CAP-ZW candidate with explicit anti-collapse controls."""

    # Rebalance away from the overly restrictive v11 robustness guard.
    lambda_robust: float = 1.00
    lambda_tail: float = 0.75
    lambda_diversity: float = 0.55
    lambda_balance: float = 0.45
    lambda_corr: float = 0.35
    lambda_entropy: float = 0.35
    lambda_uniformity: float = 0.15
    lambda_consistency: float = 0.75
    lambda_binary_collision: float = 1.05

    # New explicit objectives.
    lambda_separation_tail: float = 0.90
    lambda_entropy_floor: float = 0.60
    lambda_binary_confidence: float = 0.12
    separation_q05_target: float = 0.10
    separation_q10_target: float = 0.15
    entropy_floor_target: float = 0.82
    binary_confidence_target: float = 0.18

    # More global negative coverage.
    topk_negatives: int = 32
    memory_size: int = 8192
    memory_warmup: int = 512
    collision_power: float = 2.0
    binary_collision_target: float = 0.18
    tail_target: float = 0.30
    diversity_target: float = 0.40

    # Softer robustness constraint; still active to prevent robustness collapse.
    robustness_guard_target: float = 0.028
    robustness_guard_quantile: float = 0.80
    robustness_guard_softness: float = 0.010
    robustness_guard_lambda_init: float = 0.25
    robustness_guard_lambda_growth: float = 0.05
    robustness_guard_lambda_max: float = 1.25
    robustness_guard_tail_weight: float = 0.45
    robustness_guard_mean_weight: float = 0.10
    robustness_guard_batch_fraction: float = 0.15


def _finite_values(matrix: Tensor | None) -> Tensor:
    if matrix is None or matrix.numel() == 0:
        return torch.empty(0, device="cpu")
    values = matrix[torch.isfinite(matrix)]
    return values


def _separation_tail_loss(
    distances: Tensor | None,
    q05_target: float,
    q10_target: float,
    device: torch.device,
) -> Tensor:
    """Push the lower inter-image Hamming tail away from zero.

    Unlike a nearest-neighbor-only loss, this directly constrains the population
    tail that drives false positives and collision risk.
    """
    values = _finite_values(distances)
    if values.numel() == 0:
        return torch.zeros((), device=device)
    q05 = torch.quantile(values, 0.05)
    q10 = torch.quantile(values, 0.10)
    p05 = F.softplus((float(q05_target) - q05) / 0.025).mul(0.025)
    p10 = F.softplus((float(q10_target) - q10) / 0.025).mul(0.025)
    return p05 + 0.50 * p10


def _entropy_floor_loss(codes: Tensor, target: float) -> Tensor:
    """Penalize collapse of binary bit marginals below a target entropy."""
    p = codes.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    return F.relu(float(target) - entropy).pow(2)


def _binary_confidence_loss(codes: Tensor, target: float) -> Tensor:
    """Discourage continuous outputs from hovering around the threshold 0.5."""
    confidence = (2.0 * codes - 1.0).abs()
    return F.relu(float(target) - confidence).pow(2).mean()


def _extra_terms(
    clean: Tensor,
    attacked: Tensor,
    labels: Tensor,
    memory_codes: Tensor | None,
    memory_labels: Tensor | None,
    config: V12Config,
) -> dict[str, Tensor]:
    pair_codes = torch.cat([clean, attacked], dim=0)
    pair_labels = torch.cat([labels, labels], dim=0)
    distances = torch.cdist(pair_codes, pair_codes, p=1) / pair_codes.shape[1]
    negative = distances.masked_fill(pair_labels[:, None].eq(pair_labels[None, :]), float("inf"))
    separation = _separation_tail_loss(
        negative, config.separation_q05_target, config.separation_q10_target, clean.device
    )
    if memory_codes is not None and memory_codes.numel() > 0:
        memory_dist = torch.cdist(pair_codes, memory_codes.detach(), p=1) / pair_codes.shape[1]
        if memory_labels is not None:
            memory_dist = memory_dist.masked_fill(
                pair_labels[:, None].eq(memory_labels.detach()[None, :]), float("inf")
            )
        separation = 0.5 * (
            separation
            + _separation_tail_loss(
                memory_dist, config.separation_q05_target, config.separation_q10_target, clean.device
            )
        )
    entropy_floor = _entropy_floor_loss(pair_codes, config.entropy_floor_target)
    confidence = _binary_confidence_loss(pair_codes, config.binary_confidence_target)
    return {
        "separation_tail": separation,
        "entropy_floor": entropy_floor,
        "binary_confidence": confidence,
    }


def train_cap_zw_v12(
    model: nn.Module,
    loader,
    config: V12Config,
    checkpoint: str | None = None,
    history_path: str | None = None,
) -> list[dict[str, float]]:
    """Train the v12 candidate with the collision-tail and anti-collapse objectives."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    task_names = [
        "robustness", "tail_collision", "diversity", "balance", "decorrelation",
        "entropy_penalty", "uniformity", "consistency", "binary_collision",
        "separation_tail", "entropy_floor", "binary_confidence",
    ]
    scales = torch.tensor([
        config.lambda_robust, config.lambda_tail, config.lambda_diversity,
        config.lambda_balance, config.lambda_corr, config.lambda_entropy,
        config.lambda_uniformity, config.lambda_consistency, config.lambda_binary_collision,
        config.lambda_separation_tail, config.lambda_entropy_floor, config.lambda_binary_confidence,
    ], dtype=torch.float32, device=device)
    memory_codes = torch.empty((0, config.nbits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)
    guard_multiplier = float(config.robustness_guard_lambda_init)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in task_names}
        totals.update({
            "total": 0.0, "memory_size": 0.0, "robust_q80": 0.0,
            "robust_worst_mean": 0.0, "robust_guard": 0.0,
            "guard_multiplier": 0.0,
        })
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0

        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x = clean_x.to(device, non_blocking=True)
            attacked_x = attacked_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)
            memory_for_loss = memory_codes if memory_codes.numel() else None
            labels_for_memory = memory_labels if memory_labels.numel() else None
            pair_count = int(clean.shape[0] * 2)
            effective_topk = min(
                max(config.topk_negatives, 1),
                max(pair_count + int(memory_codes.shape[0]), 1),
            )
            progress = min(1.0, memory_codes.shape[0] / max(config.memory_warmup, 1))
            schedule = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
            active_margin = 0.18 + (float(config.margin) - 0.18) * schedule
            active_tail = 0.14 + (float(config.tail_target) - 0.14) * schedule
            active_diversity = 0.20 + (float(config.diversity_target) - 0.20) * schedule
            active_binary = 0.08 + (float(config.binary_collision_target) - 0.08) * schedule

            base = objective_terms(
                clean, attacked, labels, active_margin,
                memory_for_loss, labels_for_memory,
                config.collision_power, effective_topk,
                active_diversity, active_tail,
                config.uniformity_temperature, config.robustness_quantile,
                config.robust_target, config.robust_softness,
                active_binary,
            )
            extra = _extra_terms(
                clean, attacked, labels, memory_for_loss, labels_for_memory, config
            )
            terms = {**base, **extra}
            tasks = [terms[name] for name in task_names]
            if config.enable_mgda:
                mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            else:
                mgda = torch.ones(len(tasks), dtype=clean.dtype, device=device) / len(tasks)
            effective = mgda * scales.to(dtype=mgda.dtype)
            effective = effective / effective.sum().clamp_min(1e-12)
            base_loss = sum(weight * task for weight, task in zip(effective, tasks))

            from .v11 import _selective_robustness_guard
            guard, violation, guard_q, worst_values = _selective_robustness_guard(
                clean, attacked,
                config.robustness_guard_target,
                config.robustness_guard_quantile,
                config.robustness_guard_softness,
                config.robustness_guard_tail_weight,
                config.robustness_guard_mean_weight,
                config.robustness_guard_batch_fraction,
            )
            guard_warmup = min(1.0, (epoch + 1) / max(config.epochs * 0.50, 1.0))
            observed = float(violation)
            normalized = observed / max(config.robustness_guard_target, 1e-6)
            adaptive = 1.0 + min(1.0, max(0.0, normalized))
            guard_weight = guard_multiplier * (0.30 + 0.30 * guard_warmup) * adaptive
            loss = base_loss + guard_weight * guard

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            memory_codes, memory_labels = _update_memory(
                memory_codes, memory_labels, clean, labels, config.memory_size
            )
            if observed > 0:
                guard_multiplier = min(
                    config.robustness_guard_lambda_max,
                    guard_multiplier + config.robustness_guard_lambda_growth * min(1.0, normalized),
                )
            else:
                guard_multiplier = max(config.robustness_guard_lambda_init, guard_multiplier * 0.995)

            for name in task_names:
                totals[name] += float(terms[name].detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            totals["robust_q80"] += float(guard_q)
            totals["robust_worst_mean"] += float(worst_values.mean())
            totals["robust_guard"] += float(guard.detach())
            totals["guard_multiplier"] += guard_multiplier
            weight_totals += effective.detach().cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({name: value / denom for name, value in totals.items()})
        row.update({
            f"w_{name}": float(weight_totals[i] / denom)
            for i, name in enumerate([
                "robust", "tail", "diversity", "balance", "corr", "entropy",
                "uniformity", "consistency", "binary_collision", "separation_tail",
                "entropy_floor", "binary_confidence",
            ])
        })
        history.append(row)
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "config": config.__dict__,
                "version": CAP_ZW_V12_VERSION,
                "guard_multiplier": guard_multiplier,
            }, path)
    if history_path:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = [
    "CAP_ZW_V12_VERSION", "V12Config", "CAPZWHashNet", "PairAttackDataset",
    "train_cap_zw_v12", "_separation_tail_loss", "_entropy_floor_loss",
    "_binary_confidence_loss",
]
