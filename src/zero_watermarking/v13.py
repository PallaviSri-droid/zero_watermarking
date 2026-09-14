from __future__ import annotations

"""CAP-ZW-v13 Pareto refinement.

V13 keeps the useful collision-first ideas from v12 but changes two things exposed
by the 200-image evaluation: entropy/separation are treated as augmented constraints
rather than only MGDA tasks, and the robustness guard is scheduled from a softer
starting point toward the original robustness budget. This is a candidate family,
not a claim of superiority.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .training import CAPZWHashNet, PairAttackDataset, _update_memory, mgda_weights, objective_terms
from .v11 import _selective_robustness_guard
from .v12 import _binary_confidence_loss, _entropy_floor_loss, _separation_tail_loss

CAP_ZW_V13_VERSION = "CAP-ZW-v13"


@dataclass
class V13Config:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 2e-4
    margin: float = 0.36
    robust_target: float = 0.022
    robust_softness: float = 0.008
    nbits: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 5.0
    memory_size: int = 8192
    memory_warmup: int = 512
    mgda_steps: int = 20
    collision_power: float = 2.0
    topk_negatives: int = 32
    diversity_target: float = 0.40
    tail_target: float = 0.30
    binary_collision_target: float = 0.18
    uniformity_temperature: float = 0.08
    robustness_quantile: float = 0.80
    attack_views: int = 4

    # Base objective scales.
    lambda_robust: float = 1.00
    lambda_tail: float = 0.65
    lambda_diversity: float = 0.50
    lambda_balance: float = 0.42
    lambda_corr: float = 0.40
    lambda_entropy: float = 0.30
    lambda_uniformity: float = 0.12
    lambda_consistency: float = 0.70
    lambda_binary_collision: float = 0.95

    # Constraint-style penalties: applied outside MGDA.
    lambda_separation_constraint: float = 1.20
    lambda_entropy_constraint: float = 1.00
    lambda_confidence_constraint: float = 0.08
    separation_q05_target: float = 0.11
    separation_q10_target: float = 0.16
    entropy_floor_target: float = 0.80
    binary_confidence_target: float = 0.16
    constraint_warmup_fraction: float = 0.35

    # Softer guard early; ramps toward a useful robustness floor.
    guard_target_start: float = 0.030
    guard_target_final: float = 0.0235
    guard_quantile: float = 0.80
    guard_softness: float = 0.010
    guard_lambda_init: float = 0.20
    guard_lambda_growth: float = 0.06
    guard_lambda_max: float = 1.50
    guard_tail_weight: float = 0.50
    guard_mean_weight: float = 0.10
    guard_batch_fraction: float = 0.15


def _negative_matrix(clean: Tensor, attacked: Tensor, labels: Tensor, memory: Tensor | None, memory_labels: Tensor | None) -> tuple[Tensor, Tensor | None]:
    codes = torch.cat([clean, attacked], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    dist = torch.cdist(codes, codes, p=1) / codes.shape[1]
    neg = dist.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    mem_neg = None
    if memory is not None and memory.numel() > 0:
        mem_neg = torch.cdist(codes, memory.detach(), p=1) / codes.shape[1]
        if memory_labels is not None:
            mem_neg = mem_neg.masked_fill(labs[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
    return neg, mem_neg


def _constraint_penalty(clean: Tensor, attacked: Tensor, labels: Tensor, memory: Tensor | None, memory_labels: Tensor | None, config: V13Config, warmup: float) -> tuple[Tensor, dict[str, Tensor]]:
    neg, mem_neg = _negative_matrix(clean, attacked, labels, memory, memory_labels)
    separation = _separation_tail_loss(neg, config.separation_q05_target, config.separation_q10_target, clean.device)
    if mem_neg is not None:
        separation = 0.5 * (separation + _separation_tail_loss(mem_neg, config.separation_q05_target, config.separation_q10_target, clean.device))
    codes = torch.cat([clean, attacked], dim=0)
    entropy_floor = _entropy_floor_loss(codes, config.entropy_floor_target)
    confidence = _binary_confidence_loss(codes, config.binary_confidence_target)
    weight = float(min(max(warmup, 0.0), 1.0))
    total = weight * (
        config.lambda_separation_constraint * separation
        + config.lambda_entropy_constraint * entropy_floor
        + config.lambda_confidence_constraint * confidence
    )
    return total, {
        "separation_constraint": separation,
        "entropy_constraint": entropy_floor,
        "confidence_constraint": confidence,
    }


def train_cap_zw_v13(model: nn.Module, loader, config: V13Config, checkpoint: str | None = None, history_path: str | None = None) -> list[dict[str, float]]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    task_names = ["robustness", "tail_collision", "diversity", "balance", "decorrelation", "entropy_penalty", "uniformity", "consistency", "binary_collision"]
    scales = torch.tensor([
        config.lambda_robust, config.lambda_tail, config.lambda_diversity, config.lambda_balance,
        config.lambda_corr, config.lambda_entropy, config.lambda_uniformity, config.lambda_consistency,
        config.lambda_binary_collision,
    ], dtype=torch.float32, device=device)
    memory_codes = torch.empty((0, config.nbits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)
    guard_multiplier = config.guard_lambda_init
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        totals = {n: 0.0 for n in task_names}
        totals.update({"total": 0.0, "memory_size": 0.0, "guard_target": 0.0, "guard": 0.0, "guard_multiplier": 0.0,
                       "separation_constraint": 0.0, "entropy_constraint": 0.0, "confidence_constraint": 0.0})
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0
        epoch_progress = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
        target = config.guard_target_start + (config.guard_target_final - config.guard_target_start) * epoch_progress
        constraint_warmup = min(1.0, (epoch + 1) / max(config.epochs * config.constraint_warmup_fraction, 1.0))

        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x = clean_x.to(device, non_blocking=True)
            attacked_x = attacked_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)
            memory_for_loss = memory_codes if memory_codes.numel() else None
            memory_labels_for_loss = memory_labels if memory_labels.numel() else None
            available = clean.shape[0] * 2 + int(memory_codes.shape[0])
            effective_topk = min(max(config.topk_negatives, 1), max(available, 1))
            schedule = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
            active_margin = 0.18 + (config.margin - 0.18) * schedule
            active_tail = 0.14 + (config.tail_target - 0.14) * schedule
            active_diversity = 0.20 + (config.diversity_target - 0.20) * schedule
            active_binary = 0.08 + (config.binary_collision_target - 0.08) * schedule
            base = objective_terms(clean, attacked, labels, active_margin, memory_for_loss, memory_labels_for_loss,
                                   config.collision_power, effective_topk, active_diversity, active_tail,
                                   config.uniformity_temperature, config.robustness_quantile, config.robust_target,
                                   config.robust_softness, active_binary)
            tasks = [base[n] for n in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            effective = mgda * scales.to(dtype=mgda.dtype)
            effective = effective / effective.sum().clamp_min(1e-12)
            base_loss = sum(w * t for w, t in zip(effective, tasks))
            constraint, extra = _constraint_penalty(clean, attacked, labels, memory_for_loss, memory_labels_for_loss, config, constraint_warmup)

            guard, violation, guard_q, worst_values = _selective_robustness_guard(
                clean, attacked, target, config.guard_quantile, config.guard_softness,
                config.guard_tail_weight, config.guard_mean_weight, config.guard_batch_fraction,
            )
            observed = float(violation)
            normalized = observed / max(target, 1e-6)
            adaptive = 1.0 + min(1.0, max(0.0, normalized))
            guard_weight = guard_multiplier * (0.28 + 0.32 * epoch_progress) * adaptive
            loss = base_loss + constraint + guard_weight * guard

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            memory_codes, memory_labels = _update_memory(memory_codes, memory_labels, clean, labels, config.memory_size)
            if observed > 0.0:
                guard_multiplier = min(config.guard_lambda_max, guard_multiplier + config.guard_lambda_growth * min(1.0, normalized))
            else:
                guard_multiplier = max(config.guard_lambda_init, guard_multiplier * 0.995)

            for n in task_names:
                totals[n] += float(base[n].detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            totals["guard_target"] += target
            totals["guard"] += float(guard.detach())
            totals["guard_multiplier"] += guard_multiplier
            for n, value in extra.items():
                totals[n] += float(value.detach())
            weight_totals += effective.detach().cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({k: v / denom for k, v in totals.items()})
        row.update({f"w_{n}": float(weight_totals[i] / denom) for i, n in enumerate([
            "robust", "tail", "diversity", "balance", "corr", "entropy", "uniformity", "consistency", "binary_collision"
        ])})
        history.append(row)
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "config": config.__dict__, "version": CAP_ZW_V13_VERSION,
                        "guard_multiplier": guard_multiplier}, path)

    if history_path:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = ["CAP_ZW_V13_VERSION", "V13Config", "CAPZWHashNet", "PairAttackDataset", "train_cap_zw_v13"]
