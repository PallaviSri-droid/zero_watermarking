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


CAP_ZW_V11_VERSION = "CAP-ZW-v11"


@dataclass
class V11Config(TrainConfig):
    """CAP-ZW-v11: selective robustness protection with Pareto-preserving collision pressure."""

    # Start closer to v9 than v10, then protect only the worst robustness tail.
    lambda_binary_collision: float = 1.38
    lambda_consistency: float = 1.00
    lambda_corr: float = 0.25
    lambda_diversity: float = 0.30
    robustness_guard_target: float = 0.021
    robustness_guard_quantile: float = 0.85
    robustness_guard_softness: float = 0.008
    robustness_guard_lambda_init: float = 0.55
    robustness_guard_lambda_growth: float = 0.10
    robustness_guard_lambda_max: float = 2.50
    robustness_guard_tail_weight: float = 0.70
    robustness_guard_mean_weight: float = 0.20
    robustness_guard_batch_fraction: float = 0.25


def _selective_robustness_guard(
    clean: Tensor,
    attacked: Tensor,
    target: float,
    quantile: float,
    softness: float,
    tail_weight: float,
    mean_weight: float,
    batch_fraction: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Protect the worst robustness cases without constraining the full batch equally."""
    per_sample = (clean - attacked).abs().mean(dim=1)
    if per_sample.numel() == 0:
        zero = clean.new_zeros(())
        return zero, zero, zero, clean.new_zeros((0,))

    q = float(min(max(quantile, 0.5), 0.99))
    q_value = torch.quantile(per_sample, q)
    target_t = clean.new_tensor(float(target))
    beta = max(float(softness), 1e-4)

    # Focus explicitly on the worst quarter (or configured fraction) while retaining
    # a smooth high-quantile term so the guard does not collapse to one sample.
    fraction = float(min(max(batch_fraction, 0.05), 0.50))
    count = max(1, int(np.ceil(per_sample.numel() * fraction)))
    worst_values = torch.topk(per_sample, count, largest=True).values
    worst_penalty = F.softplus((worst_values - target_t) / beta).mul(beta).mean()
    q_penalty = F.softplus((q_value - target_t) / beta).mul(beta)
    mean_penalty = F.softplus((per_sample.mean() - target_t) / beta).mul(beta)

    penalty = float(tail_weight) * (worst_penalty + 0.50 * q_penalty) + float(mean_weight) * mean_penalty
    violation = (
        float(tail_weight) * (F.relu(worst_values - target_t).mean() + 0.50 * F.relu(q_value - target_t))
        + float(mean_weight) * F.relu(per_sample.mean() - target_t)
    )
    return penalty, violation.detach(), q_value.detach(), worst_values.detach()


def train_cap_zw_v11(
    model: nn.Module,
    loader,
    config: V11Config,
    checkpoint: str | None = None,
    history_path: str | None = None,
) -> list[dict[str, float]]:
    """Train CAP-ZW-v11 using selective robustness protection instead of a global guard."""
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
            "robust_guard", "robust_guard_violation", "robust_q85", "robust_worst_mean",
            "guard_multiplier",
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
                clean, attacked, labels, active_margin,
                memory_codes if progress > 0 else None,
                memory_labels if progress > 0 else None,
                config.collision_power, config.topk_negatives,
                active_diversity, active_tail,
                config.uniformity_temperature, config.robustness_quantile,
                config.robust_target, config.robust_softness,
                active_binary,
            )
            tasks = [terms[name] for name in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            scale = torch.tensor([
                config.lambda_robust, config.lambda_tail, config.lambda_diversity,
                config.lambda_balance, config.lambda_corr, config.lambda_entropy,
                config.lambda_uniformity, config.lambda_consistency, config.lambda_binary_collision,
            ], dtype=mgda.dtype, device=device)
            effective = mgda * scale
            effective = effective / effective.sum().clamp_min(1e-12)
            base_loss = sum(weight * task for weight, task in zip(effective, tasks))

            guard, violation, guard_q, worst_values = _selective_robustness_guard(
                clean,
                attacked,
                config.robustness_guard_target,
                config.robustness_guard_quantile,
                config.robustness_guard_softness,
                config.robustness_guard_tail_weight,
                config.robustness_guard_mean_weight,
                config.robustness_guard_batch_fraction,
            )
            # Guard ramps in slowly and only becomes material when the protected
            # robustness tail exceeds the budget. This avoids the v10 collapse in
            # entropy/balance caused by globally strong robustness pressure.
            guard_warmup = min(1.0, (epoch + 1) / max(config.epochs * 0.50, 1.0))
            observed_violation = float(violation.detach())
            normalized_violation = observed_violation / max(config.robustness_guard_target, 1e-6)
            adaptive = 1.0 + min(1.0, max(0.0, normalized_violation))
            guard_weight = guard_multiplier * (0.35 + 0.35 * guard_warmup) * adaptive
            loss = base_loss + guard_weight * guard

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            memory_codes, memory_labels = _update_memory(
                memory_codes, memory_labels, clean, labels, config.memory_size
            )

            if observed_violation > 0.0:
                guard_multiplier = min(
                    float(config.robustness_guard_lambda_max),
                    guard_multiplier + float(config.robustness_guard_lambda_growth) * min(1.0, normalized_violation),
                )
            else:
                guard_multiplier = max(
                    float(config.robustness_guard_lambda_init),
                    guard_multiplier * 0.995,
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
            totals["robust_guard_violation"] += observed_violation
            totals["robust_q85"] += float(guard_q)
            totals["robust_worst_mean"] += float(worst_values.mean())
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
                "version": CAP_ZW_V11_VERSION,
                "guard_multiplier": guard_multiplier,
            }, path)

    if history_path:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = [
    "CAP_ZW_V11_VERSION",
    "V11Config",
    "CAPZWHashNet",
    "PairAttackDataset",
    "train_cap_zw_v11",
    "_selective_robustness_guard",
]
