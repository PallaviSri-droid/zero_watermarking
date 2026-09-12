from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from .attacks import ATTACKS
from .learned import HashEncoder


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 6e-4
    weight_decay: float = 2e-4
    margin: float = 0.38
    robust_target: float = 0.022
    robust_softness: float = 0.01
    lambda_robust: float = 1.35
    lambda_tail: float = 0.85
    lambda_diversity: float = 0.45
    lambda_balance: float = 0.30
    lambda_corr: float = 0.12
    lambda_entropy: float = 0.12
    lambda_uniformity: float = 0.18
    lambda_consistency: float = 0.65
    nbits: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 5.0
    memory_size: int = 2048
    memory_warmup: int = 256
    mgda_steps: int = 20
    collision_power: float = 2.0
    topk_negatives: int = 8
    diversity_target: float = 0.40
    tail_target: float = 0.30
    uniformity_temperature: float = 0.08
    robustness_quantile: float = 0.80
    attack_views: int = 3


class PairAttackDataset(Dataset):
    """Deterministic multi-view clean/attacked pairs for reproducible training."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_names: Iterable[str] = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound"),
        attack_views: int = 3,
    ) -> None:
        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)
        if images.ndim != 3:
            raise ValueError("images must have shape [N,H,W]")
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.attack_names = tuple(attack_names)
        if not self.attack_names or any(name not in ATTACKS or name == "clean" for name in self.attack_names):
            raise ValueError("attack_names must contain valid non-clean attack names")
        self.images = np.clip(images, 0.0, 1.0)
        self.labels = labels.astype(np.int64)
        self.attack_views = max(1, int(attack_views))

    def __len__(self) -> int:
        return int(len(self.images))

    def _attack(self, clean: np.ndarray, index: int, view: int) -> np.ndarray:
        name = self.attack_names[(index * self.attack_views + view) % len(self.attack_names)]
        seed = index * 97 + view
        defaults = {
            "gaussian_noise": {"sigma": 0.03 + 0.01 * (view % 2), "seed": seed},
            "salt_pepper": {"amount": 0.01 + 0.005 * (view % 2), "seed": seed},
            "gaussian_blur": {"sigma": 1.0 + 0.5 * (view % 2)},
            "median_blur": {"size": 3},
            "jpeg": {"quality": 70 if view % 2 == 0 else 40},
            "brightness": {"factor": 1.10 if view % 2 == 0 else 0.92},
            "contrast": {"factor": 1.15 if view % 2 == 0 else 0.88},
            "rotation": {"degrees": 5.0 if view % 2 == 0 else 10.0},
            "crop_resize": {"fraction": 0.05},
            "translation": {"pixels": 3},
            "compound": {"seed": seed},
        }
        return ATTACKS[name](clean, **defaults.get(name, {}))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        clean = self.images[index]
        view = index % self.attack_views
        attack_name = self.attack_names[(index * self.attack_views + view) % len(self.attack_names)]
        attacked = self._attack(clean, index, view)
        return (
            torch.from_numpy(clean[None]),
            torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None]),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            attack_name,
        )


def _normalized_hamming(z1: Tensor, z2: Tensor) -> Tensor:
    return torch.cdist(z1, z2, p=1) / z1.shape[1]


def _finite_topk(values: Tensor, k: int) -> list[Tensor]:
    rows: list[Tensor] = []
    for row in values:
        finite = row[torch.isfinite(row)]
        if finite.numel() == 0:
            rows.append(row.new_zeros(1))
            continue
        kk = min(max(int(k), 1), int(finite.numel()))
        rows.append(torch.topk(finite, kk, largest=False).values)
    return rows


def _soft_negative_loss(distances: Tensor | None, margin: float, topk: int, temperature: float, device: torch.device) -> Tensor:
    if distances is None or distances.numel() == 0:
        return torch.zeros((), device=device)
    rows = _finite_topk(distances, topk)
    tau = max(float(temperature), 1e-4)
    return torch.stack([F.softplus((float(margin) - r) / tau).mul(tau).mean() for r in rows]).mean()


def _tail_collision_loss(distances: Tensor | None, target: float, topk: int, power: float, device: torch.device) -> Tensor:
    if distances is None or distances.numel() == 0:
        return torch.zeros((), device=device)
    rows = _finite_topk(distances, topk)
    penalties = [F.relu(float(target) - r.min()).pow(max(float(power), 1.0)) for r in rows]
    return torch.stack(penalties).mean()


def _uniformity_loss(distances: Tensor | None, temperature: float, topk: int, device: torch.device) -> Tensor:
    if distances is None or distances.numel() == 0:
        return torch.zeros((), device=device)
    rows = _finite_topk(distances, topk)
    tau = max(float(temperature), 1e-4)
    return torch.stack([torch.exp(-r / tau).mean() for r in rows]).mean()


def _robust_floor_loss(clean: Tensor, attacked: Tensor, target: float, softness: float) -> Tensor:
    per_sample = (clean - attacked).abs().mean(dim=1)
    target_t = clean.new_tensor(float(target))
    beta = max(float(softness), 1e-4)
    return F.softplus((per_sample - target_t) / beta).mul(beta).mean()


def _consistency_loss(clean: Tensor, attacked: Tensor) -> Tensor:
    # Symmetric bitwise consistency keeps clean/attacked views aligned without forcing every
    # representation toward an all-zero/all-one solution.
    p = clean.clamp(1e-5, 1.0 - 1e-5)
    q = attacked.clamp(1e-5, 1.0 - 1e-5)
    kl_pq = p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q))
    kl_qp = q * (torch.log(q) - torch.log(p)) + (1.0 - q) * (torch.log1p(-q) - torch.log1p(-p))
    return 0.5 * (kl_pq + kl_qp).mean()


def objective_terms(
    clean: Tensor,
    attacked: Tensor,
    labels: Tensor,
    margin: float,
    memory_codes: Tensor | None = None,
    memory_labels: Tensor | None = None,
    collision_power: float = 2.0,
    topk_negatives: int = 8,
    diversity_target: float = 0.40,
    tail_target: float = 0.30,
    uniformity_temperature: float = 0.08,
    robustness_quantile: float = 0.80,
    robust_target: float = 0.022,
    robust_softness: float = 0.01,
) -> Dict[str, Tensor]:
    """CAP-ZW v6: constrained robustness + tail-aware collision optimization."""
    if clean.ndim != 2 or attacked.ndim != 2 or clean.shape != attacked.shape:
        raise ValueError("clean and attacked must both have shape [B, nbits]")

    pair_codes = torch.cat([clean, attacked], dim=0)
    pair_labels = torch.cat([labels, labels], dim=0)

    per_sample_robust = (clean - attacked).abs().mean(dim=1)
    q = float(min(max(robustness_quantile, 0.5), 1.0))
    q_value = torch.quantile(per_sample_robust.detach(), q)
    robust_weights = torch.sigmoid((per_sample_robust - q_value) / 0.012) + 0.30
    tail_weighted = (per_sample_robust * robust_weights).mean() / robust_weights.mean().clamp_min(1e-6)
    floor = _robust_floor_loss(clean, attacked, robust_target, robust_softness)
    robustness = tail_weighted + 0.75 * floor
    consistency = _consistency_loss(clean, attacked)

    distances = _normalized_hamming(pair_codes, pair_codes)
    same = pair_labels[:, None].eq(pair_labels[None, :])
    pair_neg = distances.masked_fill(same, float("inf"))

    memory_neg = None
    if memory_codes is not None and memory_codes.numel() > 0:
        memory_neg = _normalized_hamming(pair_codes, memory_codes.detach())
        if memory_labels is not None:
            memory_neg = memory_neg.masked_fill(pair_labels[:, None].eq(memory_labels.detach()[None, :]), float("inf"))

    soft_batch = _soft_negative_loss(pair_neg, margin, topk_negatives, uniformity_temperature, clean.device)
    soft_memory = _soft_negative_loss(memory_neg, margin, topk_negatives, uniformity_temperature, clean.device) if memory_neg is not None else torch.zeros((), device=clean.device)
    discrimination = (soft_batch + soft_memory) / (2.0 if memory_neg is not None else 1.0)

    batch_tail = _tail_collision_loss(pair_neg, tail_target, topk_negatives, collision_power, clean.device)
    memory_tail = _tail_collision_loss(memory_neg, tail_target, topk_negatives, collision_power, clean.device) if memory_neg is not None else torch.zeros((), device=clean.device)
    tail_collision = (batch_tail + memory_tail) / (2.0 if memory_neg is not None else 1.0) + 0.15 * discrimination

    cross_min = pair_neg.min(dim=1).values
    if memory_neg is not None:
        cross_min = torch.minimum(cross_min, memory_neg.min(dim=1).values)
    valid = torch.isfinite(cross_min)
    diversity = F.relu(float(diversity_target) - cross_min[valid]).pow(2).mean() if valid.any() else torch.zeros((), device=clean.device)

    sources = [pair_neg] + ([memory_neg] if memory_neg is not None else [])
    uniformity = torch.stack([_uniformity_loss(s, uniformity_temperature, topk_negatives, clean.device) for s in sources]).mean()

    combined = torch.cat([clean, attacked], dim=0)
    p = combined.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    balance = (p - 0.5).pow(2).mean()
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()

    centered = combined - combined.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(2e-3)
    normalized = centered / std
    corr_mat = (normalized.T @ normalized) / max(combined.shape[0], 1)
    eye = torch.eye(corr_mat.shape[0], device=combined.device, dtype=combined.dtype)
    decorrelation = ((corr_mat - eye) * (1.0 - eye)).pow(2).mean()

    return {
        "robustness": robustness,
        "tail_collision": tail_collision,
        "diversity": diversity,
        "balance": balance,
        "decorrelation": decorrelation,
        "entropy_penalty": 1.0 - entropy,
        "uniformity": uniformity,
        "consistency": consistency,
    }


def _flatten_gradients(grads: list[Tensor | None], params: list[Tensor]) -> Tensor:
    return torch.cat([torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1) for grad, param in zip(grads, params)])


def _project_simplex(x: Tensor) -> Tensor:
    if x.numel() == 1:
        return torch.ones_like(x)
    u, _ = torch.sort(x, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    rho_candidates = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    cond = u - cssv / rho_candidates > 0
    rho = torch.where(cond)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(x - theta, min=0.0)


def mgda_weights(losses: list[Tensor], model: nn.Module, steps: int = 20) -> Tensor:
    if not losses:
        raise ValueError("losses must not be empty")
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = []
    for loss in losses:
        if not torch.isfinite(loss.detach()):
            vectors.append(torch.zeros(sum(p.numel() for p in params), device=loss.device, dtype=loss.dtype))
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors.append(_flatten_gradients(grads, params))
    G = torch.stack(vectors, dim=1)
    gram = G.T @ G
    count = len(losses)
    weights = torch.full((count,), 1.0 / count, dtype=G.dtype, device=G.device)
    spectral = float(torch.linalg.matrix_norm(gram, ord=2).detach().cpu()) if count > 1 else 1.0
    step = min(0.20, 1.0 / max(2.0 * spectral, 1e-6))
    for _ in range(max(1, steps)):
        weights = _project_simplex(weights - step * (2.0 * gram @ weights))
    return weights.detach()


def _update_memory(memory_codes: Tensor, memory_labels: Tensor, new_codes: Tensor, new_labels: Tensor, max_size: int):
    if max_size <= 0:
        return memory_codes[:0], memory_labels[:0]
    codes = torch.cat([memory_codes, new_codes.detach()], dim=0)
    labels = torch.cat([memory_labels, new_labels.detach()], dim=0)
    if codes.shape[0] > max_size:
        codes = codes[-max_size:]
        labels = labels[-max_size:]
    return codes, labels


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    """Train CAP-ZW v6 with robustness-constrained tail collision optimization."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    task_names = ["robustness", "tail_collision", "diversity", "balance", "decorrelation", "entropy_penalty", "uniformity", "consistency"]
    memory_codes = torch.empty((0, config.nbits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)

    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in task_names + ["total", "memory_size", "active_margin", "active_tail", "active_diversity", "robust_target", "progress"]}
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0
        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x, attacked_x, labels = clean_x.to(device, non_blocking=True), attacked_x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            clean, attacked = model(clean_x, hard=False), model(attacked_x, hard=False)

            progress = min(1.0, memory_codes.shape[0] / max(config.memory_warmup, 1))
            schedule = min(1.0, (epoch + 1) / max(config.epochs * 0.6, 1.0))
            active_margin = 0.22 + (float(config.margin) - 0.22) * schedule
            active_tail = 0.18 + (float(config.tail_target) - 0.18) * schedule
            active_diversity = 0.24 + (float(config.diversity_target) - 0.24) * schedule
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
            ], dtype=mgda.dtype, device=device)
            effective = mgda * scale
            effective = effective / effective.sum().clamp_min(1e-12)
            loss = sum(w * t for w, t in zip(effective, tasks))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            memory_codes, memory_labels = _update_memory(memory_codes, memory_labels, clean, labels, config.memory_size)

            for key, value in terms.items():
                totals[key] += float(value.detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            totals["active_margin"] += active_margin
            totals["active_tail"] += active_tail
            totals["active_diversity"] += active_diversity
            totals["robust_target"] += config.robust_target
            totals["progress"] += progress
            weight_totals += effective.detach().cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({key: value / denom for key, value in totals.items()})
        row.update({f"w_{name}": float(weight_totals[i] / denom) for i, name in enumerate(["robust", "tail", "diversity", "balance", "corr", "entropy", "uniformity", "consistency"])})
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
                "version": "CAP-ZW-v6",
            }, path)
    return history


class CAPZWHashNet(HashEncoder):
    """CAP-ZW v6: BEMQ encoder with robustness constraint and tail-aware collision control."""

    def __init__(self, nbits: int = 256, base_channels: int = 32):
        super().__init__(nbits=nbits, base_channels=base_channels, use_bemq=True)
