from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .training import (
    CAPZWHashNet,
    PairAttackDataset,
    TrainConfig,
    _update_memory,
    mgda_weights,
    objective_terms,
)


CAP_ZW_V10_VERSION = "CAP-ZW-v10"


@dataclass
class V10Config(TrainConfig):
    """CAP-ZW-v10: augmented-Lagrangian robustness guard around the v9 objective."""

    lambda_binary_collision: float = 1.20
    lambda_consistency: float = 1.15
    lambda_corr: float = 0.26
    lambda_diversity: float = 0.28
    robustness_guard_target: float = 0.019
    robustness_guard_quantile: float = 0.90
    robustness_guard_softness: float = 0.006
    robustness_guard_lambda_init: float = 1.50
    robustness_guard_lambda_growth: float = 0.25
    robustness_guard_lambda_max: float = 6.0
    robustness_guard_mean_weight: float = 0.50


def _robustness_guard(
    clean: Tensor,
    attacked: Tensor,
    target: float,
    quantile: float,
    softness: float,
    mean_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiable robustness budget using mean + high-quantile violations."""
    per_sample = (clean - attacked).abs().mean(dim=1)
    q = float(min(max(quantile, 0.5), 0.99))
    q_value = torch.quantile(per_sample, q)
    mean_value = per_sample.mean()
    beta = max(float(softness), 1e-4)
    target_t = clean.new_tensor(float(target))
    q_violation = F.softplus((q_value - target_t) / beta).mul(beta)
    mean_violation = F.softplus((mean_value - target_t) / beta).mul(beta)
    penalty = q_violation + float(mean_weight) * mean_violation
    violation = F.relu(q_value - target_t) + float(mean_weight) * F.relu(mean_value - target_t)
    return penalty, violation.detach(), q_value.detach()


def train_cap_zw_v10(
    model: nn.Module,
    loader,
    config: V10Config,
    checkpoint: str | None = None,
    history_path: str | None = None,
) -> list[dict[str, float]]:
    """Train CAP-ZW-v10 with an adaptive robustness constraint.

    The base v9 multi-objective loss is preserved. An augmented-Lagrangian guard
    adds pressure when the high-quantile attacked/clean code drift exceeds the
    robustness budget, allowing collision objectives to dominate again once the
    model is back inside the budget.
    """
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    task_names = [
        "robustness", "tail_collision", "diversity", "balance", "decorrelation",
        "entropy_penalty", "uniformity", "consistency", "binary_collision",
    ]
    memory_codes = torch.empty((0, config.nbits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)
    guard_multiplier = float(config.robustness_guard_lambda_init)

    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in task_names + [
            "total", "memory_size", "active_margin", "active_tail", "active_diversity",
            "binary_target", "robust_target", "collision_gate", "progress",
            "robust_guard", "robust_guard_violation", "robust_q90", "guard_multiplier",
        ]}
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0

        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x = clean_x.to(device, non_blocking=True)
            attacked_x = attacked_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)

            progress = min(1.0, memory_codes.shape[0] / max(config.memory_warmup, 1))
            schedule = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
            active_margin = 0.18 + (float(config.margin) - 0.18) * schedule
            active_tail = 0.14 + (float(config.tail_target) - 0.14) * schedule
            active_diversity = 0.20 + (float(config.diversity_target) - 0.20) * schedule
            active_binary = 0.08 + (float(config.binary_collision_target) - 0.08) * schedule

            terms = objective_terms(
                clean,
                attacked,
                labels,
                active_margin,
                memory_codes if progress > 0 else None,
                memory_labels if progress > 0 else None,
                config.collision_power,
                config.topk_negatives,
                active_diversity,
                active_tail,
                config.uniformity_temperature,
                config.robustness_quantile,
                config.robust_target,
                config.robust_softness,
                active_binary,
            )
            tasks = [terms[name] for name in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            scale = torch.tensor([
                config.lambda_robust,
                config.lambda_tail,
                config.lambda_diversity,
                config.lambda_balance,
                config.lambda_corr,
                config.lambda_entropy,
                config.lambda_uniformity,
                config.lambda_consistency,
                config.lambda_binary_collision,
            ], dtype=mgda.dtype, device=device)
            effective = mgda * scale
            effective = effective / effective.sum().clamp_min(1e-12)
            base_loss = sum(weight * task for weight, task in zip(effective, tasks))

            guard, violation, guard_q = _robustness_guard(
                clean,
                attacked,
                config.robustness_guard_target,
                config.robustness_guard_quantile,
                config.robustness_guard_softness,
                config.robustness_guard_mean_weight,
            )
            guard_warmup = min(1.0, (epoch + 1) / max(config.epochs * 0.40, 1.0))
            guard_weight = guard_multiplier * (0.50 + 0.50 * guard_warmup)
            loss = base_loss + guard_weight * guard

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            memory_codes, memory_labels = _update_memory(
                memory_codes, memory_labels, clean, labels, config.memory_size
            )

            violation_scalar = float(violation.cpu())
            if violation_scalar > 0.0:
                guard_multiplier = min(
                    float(config.robustness_guard_lambda_max),
                    guard_multiplier + float(config.robustness_guard_lambda_growth) * violation_scalar / max(config.robustness_guard_target, 1e-6),
                )
            else:
                guard_multiplier = max(
                    float(config.robustness_guard_lambda_init),
                    guard_multiplier * 0.997,
                )

            for key, value in terms.items():
                totals[key] += float(value.detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            totals["active_margin"] += active_margin
            totals["active_tail"] += active_tail
            totals["active_diversity"] += active_diversity
            totals["binary_target"] += active_binary
            totals["robust_target"] += config.robust_target
            totals["progress"] += progress
            totals["robust_guard"] += float(guard.detach())
            totals["robust_guard_violation"] += violation_scalar
            totals["robust_q90"] += float(guard_q)
            totals["guard_multiplier"] += guard_multiplier
            batch_violation = float(torch.relu((clean - attacked).abs().mean().detach() - config.robust_target))
            totals["collision_gate"] += float(
                torch.sigmoid(-torch.tensor(batch_violation, device=device) / max(config.robust_softness, 1e-4)).clamp(0.35, 1.0)
            )
            weight_totals += effective.detach().cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({key: value / denom for key, value in totals.items()})
        row.update({
            f"w_{name}": float(weight_totals[i] / denom)
            for i, name in enumerate([
                "robust", "tail", "diversity", "balance", "corr", "entropy",
                "uniformity", "consistency", "binary_collision",
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
                "version": CAP_ZW_V10_VERSION,
                "guard_multiplier": guard_multiplier,
            }, path)

    if history_path:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = [
    "CAP_ZW_V10_VERSION",
    "V10Config",
    "CAPZWHashNet",
    "PairAttackDataset",
    "train_cap_zw_v10",
]