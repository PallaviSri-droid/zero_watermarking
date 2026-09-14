from __future__ import annotations

"""CAP-ZW-v14: collision-safe Pareto candidate.

V14 is a controlled refinement of v13 motivated by the 200-image result: v13
reduced collisions but over-relaxed robustness and still produced low measured
bit entropy. V14 adds forward-hard (STE) entropy/balance constraints and a
worst-case robustness cap, while keeping v13's lower-tail separation.
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

CAP_ZW_V14_VERSION = "CAP-ZW-v14"


@dataclass
class V14Config:
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
    topk_negatives: int = 24
    diversity_target: float = 0.40
    tail_target: float = 0.30
    binary_collision_target: float = 0.18
    uniformity_temperature: float = 0.08
    robustness_quantile: float = 0.80

    lambda_robust: float = 1.10
    lambda_tail: float = 0.70
    lambda_diversity: float = 0.55
    lambda_balance: float = 0.55
    lambda_corr: float = 0.42
    lambda_entropy: float = 0.30
    lambda_uniformity: float = 0.14
    lambda_consistency: float = 0.80
    lambda_binary_collision: float = 0.90

    lambda_separation_constraint: float = 1.05
    lambda_entropy_constraint: float = 1.40
    lambda_balance_constraint: float = 0.55
    lambda_confidence_constraint: float = 0.06
    separation_q05_target: float = 0.10
    separation_q10_target: float = 0.16
    entropy_floor_target: float = 0.82
    balance_abs_target: float = 0.12
    binary_confidence_target: float = 0.16
    constraint_warmup_fraction: float = 0.30

    guard_target_start: float = 0.026
    guard_target_final: float = 0.0225
    guard_quantile: float = 0.80
    guard_softness: float = 0.009
    guard_lambda_init: float = 0.32
    guard_lambda_growth: float = 0.06
    guard_lambda_max: float = 1.60
    guard_tail_weight: float = 0.65
    guard_mean_weight: float = 0.18
    guard_batch_fraction: float = 0.20
    max_robustness_target: float = 0.080
    max_robustness_weight: float = 0.65


def _hard_entropy_and_balance(codes: Tensor, entropy_target: float, balance_target: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate entropy/balance on hard forward bits with STE gradients."""
    hard = (codes >= 0.5).to(codes.dtype).detach() - codes.detach() + codes
    p = hard.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    entropy_penalty = F.relu(float(entropy_target) - entropy).pow(2)
    balance_penalty = F.relu(balance - float(balance_target)).pow(2)
    return entropy_penalty, balance_penalty, entropy.detach(), balance.detach()


def _robustness_cap(clean: Tensor, attacked: Tensor, target: float) -> Tensor:
    d = (clean - attacked).abs().mean(dim=1)
    return F.relu(d.max() - float(target)).pow(2)


def _constraint_penalty(clean: Tensor, attacked: Tensor, labels: Tensor, memory: Tensor | None,
                        memory_labels: Tensor | None, config: V14Config, warmup: float) -> tuple[Tensor, dict[str, Tensor]]:
    codes = torch.cat([clean, attacked], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    distances = torch.cdist(codes, codes, p=1) / codes.shape[1]
    neg = distances.masked_fill(labs[:, None].eq(labs[None, :]), float("inf"))
    separation = _separation_tail_loss(neg, config.separation_q05_target, config.separation_q10_target, clean.device)
    if memory is not None and memory.numel() > 0:
        md = torch.cdist(codes, memory.detach(), p=1) / codes.shape[1]
        if memory_labels is not None:
            md = md.masked_fill(labs[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
        separation = 0.5 * (separation + _separation_tail_loss(md, config.separation_q05_target, config.separation_q10_target, clean.device))
    entropy_soft = _entropy_floor_loss(codes, config.entropy_floor_target)
    entropy_hard, balance_hard, entropy_obs, balance_obs = _hard_entropy_and_balance(codes, config.entropy_floor_target, config.balance_abs_target)
    confidence = _binary_confidence_loss(codes, config.binary_confidence_target)
    cap = _robustness_cap(clean, attacked, config.max_robustness_target)
    w = float(min(max(warmup, 0.0), 1.0))
    total = w * (
        config.lambda_separation_constraint * separation
        + config.lambda_entropy_constraint * (0.35 * entropy_soft + 0.65 * entropy_hard)
        + config.lambda_balance_constraint * balance_hard
        + config.lambda_confidence_constraint * confidence
        + config.max_robustness_weight * cap
    )
    return total, {
        "separation_constraint": separation,
        "entropy_constraint": 0.35 * entropy_soft + 0.65 * entropy_hard,
        "balance_constraint": balance_hard,
        "confidence_constraint": confidence,
        "robustness_cap": cap,
        "entropy_observed": entropy_obs,
        "balance_observed": balance_obs,
    }


def train_cap_zw_v14(model: nn.Module, loader, config: V14Config, checkpoint: str | None = None,
                     history_path: str | None = None) -> list[dict[str, float]]:
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
        totals.update({"total": 0.0, "memory_size": 0.0, "guard_target": 0.0, "guard": 0.0,
                       "guard_multiplier": 0.0, "separation_constraint": 0.0,
                       "entropy_constraint": 0.0, "balance_constraint": 0.0,
                       "confidence_constraint": 0.0, "robustness_cap": 0.0,
                       "entropy_observed": 0.0, "balance_observed": 0.0})
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0
        epoch_progress = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
        guard_target = config.guard_target_start + (config.guard_target_final - config.guard_target_start) * epoch_progress
        constraint_warmup = min(1.0, (epoch + 1) / max(config.epochs * config.constraint_warmup_fraction, 1.0))

        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x, attacked_x, labels = clean_x.to(device), attacked_x.to(device), labels.to(device)
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)
            memory_for_loss = memory_codes if memory_codes.numel() else None
            labels_for_memory = memory_labels if memory_labels.numel() else None
            available = clean.shape[0] * 2 + int(memory_codes.shape[0])
            topk = min(max(config.topk_negatives, 1), max(available, 1))
            schedule = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
            active_margin = 0.18 + (config.margin - 0.18) * schedule
            active_tail = 0.14 + (config.tail_target - 0.14) * schedule
            active_diversity = 0.20 + (config.diversity_target - 0.20) * schedule
            active_binary = 0.08 + (config.binary_collision_target - 0.08) * schedule
            base = objective_terms(clean, attacked, labels, active_margin, memory_for_loss, labels_for_memory,
                                   config.collision_power, topk, active_diversity, active_tail,
                                   config.uniformity_temperature, config.robustness_quantile,
                                   config.robust_target, config.robust_softness, active_binary)
            tasks = [base[n] for n in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            effective = mgda * scales.to(dtype=mgda.dtype)
            effective = effective / effective.sum().clamp_min(1e-12)
            base_loss = sum(w * t for w, t in zip(effective, tasks))
            constraint, extra = _constraint_penalty(clean, attacked, labels, memory_for_loss, labels_for_memory, config, constraint_warmup)
            guard, violation, guard_q, worst = _selective_robustness_guard(
                clean, attacked, guard_target, config.guard_quantile, config.guard_softness,
                config.guard_tail_weight, config.guard_mean_weight, config.guard_batch_fraction)
            observed = float(violation)
            normalized = observed / max(guard_target, 1e-6)
            adaptive = 1.0 + min(1.0, max(0.0, normalized))
            guard_weight = guard_multiplier * (0.30 + 0.25 * epoch_progress) * adaptive
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
            totals["guard_target"] += guard_target
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
            "robust", "tail", "diversity", "balance", "corr", "entropy", "uniformity", "consistency", "binary_collision"])})
        history.append(row)
        if checkpoint:
            path = Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "config": config.__dict__, "version": CAP_ZW_V14_VERSION,
                        "guard_multiplier": guard_multiplier}, path)
    if history_path:
        path = Path(history_path); path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = ["CAP_ZW_V14_VERSION", "V14Config", "CAPZWHashNet", "PairAttackDataset", "train_cap_zw_v14"]
