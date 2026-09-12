from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .attacks import ATTACKS
from .learned import HashEncoder


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    margin: float = 0.30
    lambda_robust: float = 1.0
    lambda_collision: float = 0.7
    lambda_balance: float = 0.30
    lambda_corr: float = 0.15
    lambda_entropy: float = 0.15
    lambda_diversity: float = 0.30
    nbits: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 5.0
    memory_size: int = 2048
    memory_warmup: int = 256
    mgda_steps: int = 25
    collision_power: float = 2.0
    topk_negatives: int = 8
    diversity_target: float = 0.22
    temperature: float = 0.08


class PairAttackDataset(Dataset):
    """Deterministic clean/attacked pairs for reproducible multi-attack training."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, attack_names: Iterable[str] = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound")) -> None:
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

    def __len__(self) -> int:
        return len(self.images)

    def _attack(self, clean: np.ndarray, index: int) -> np.ndarray:
        name = self.attack_names[index % len(self.attack_names)]
        defaults = {
            "gaussian_noise": {"sigma": 0.03, "seed": index},
            "salt_pepper": {"amount": 0.01, "seed": index},
            "gaussian_blur": {"sigma": 1.0},
            "median_blur": {"size": 3},
            "jpeg": {"quality": 70},
            "brightness": {"factor": 1.10},
            "contrast": {"factor": 1.15},
            "rotation": {"degrees": 5.0},
            "crop_resize": {"fraction": 0.05},
            "translation": {"pixels": 3},
            "compound": {"seed": index},
        }
        return ATTACKS[name](clean, **defaults.get(name, {}))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        clean = self.images[index]
        attack_name = self.attack_names[index % len(self.attack_names)]
        attacked = self._attack(clean, index)
        return (torch.from_numpy(clean[None]), torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None]), torch.tensor(int(self.labels[index]), dtype=torch.long), attack_name)


def _pairwise_hamming(z: Tensor) -> Tensor:
    return torch.cdist(z, z, p=1) / z.shape[1]


def _soft_negative_loss(distances: Tensor, margin: float, topk: int, temperature: float) -> Tensor:
    if distances.numel() == 0:
        return distances.new_tensor(0.0)
    finite_rows = torch.isfinite(distances).any(dim=1)
    if not finite_rows.any():
        return distances.new_tensor(0.0)
    distances = distances[finite_rows]
    distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, 1.0))
    k = min(max(int(topk), 1), distances.shape[1])
    values = torch.topk(distances, k=k, dim=1, largest=False).values
    tau = max(float(temperature), 1e-4)
    # Softplus gives non-zero gradients even for moderately separated pairs.
    return (torch.nn.functional.softplus((margin - values) / tau) * tau).mean()


def objective_terms(clean: Tensor, attacked: Tensor, labels: Tensor, margin: float, memory_codes: Tensor | None = None, memory_labels: Tensor | None = None, collision_power: float = 2.0, topk_negatives: int = 8, diversity_target: float = 0.22, temperature: float = 0.08) -> Dict[str, Tensor]:
    """Balanced collision-memory objective.

    The loss deliberately separates robustness from global code-space quality:
    robustness keeps attacked views together, while top-k memory negatives,
    diversity, balance, entropy and decorrelation prevent representation collapse.
    """
    robustness = (clean - attacked).abs().mean()

    distances = _pairwise_hamming(clean)
    same = labels[:, None].eq(labels[None, :])
    batch_neg = distances.masked_fill(same, float("inf"))
    batch_loss = _soft_negative_loss(batch_neg, margin, topk_negatives, temperature)

    memory_loss = clean.new_tensor(0.0)
    cross_min = batch_neg.min(dim=1).values
    if memory_codes is not None and memory_codes.numel() > 0:
        memory_codes = memory_codes.detach()
        memory_labels = memory_labels.detach() if memory_labels is not None else None
        md = torch.cdist(clean, memory_codes, p=1) / clean.shape[1]
        if memory_labels is not None:
            md = md.masked_fill(labels[:, None].eq(memory_labels[None, :]), float("inf"))
        memory_loss = _soft_negative_loss(md, margin, topk_negatives, temperature)
        memory_min = md.min(dim=1).values
        cross_min = torch.minimum(cross_min, memory_min)

    collision = (batch_loss + memory_loss) / (2.0 if memory_loss.detach().item() > 0 else 1.0)
    discrimination = collision + 0.35 * collision.pow(max(collision_power, 1.0))

    # Balance/entropy are computed on clean and attacked codes to avoid learning
    # a balanced clean code that becomes biased under attacks.
    combined = torch.cat([clean, attacked], dim=0)
    p = combined.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    balance = (p - 0.5).pow(2).mean()
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()

    centered = combined - combined.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = centered / std
    corr_mat = (normalized.T @ normalized) / max(combined.shape[0], 1)
    eye = torch.eye(corr_mat.shape[0], device=combined.device, dtype=combined.dtype)
    decorrelation = ((corr_mat - eye) * (1.0 - eye)).pow(2).mean()

    valid = torch.isfinite(cross_min)
    diversity = torch.relu(diversity_target - cross_min[valid]).pow(2).mean() if valid.any() else clean.new_tensor(0.0)

    return {"robustness": robustness, "discrimination": discrimination, "balance": balance, "decorrelation": decorrelation, "entropy_penalty": 1.0 - entropy, "diversity": diversity}


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


def mgda_weights(losses: list[Tensor], model: nn.Module, steps: int = 25) -> Tensor:
    """Minimum-norm multi-gradient simplex weighting."""
    if not losses:
        raise ValueError("losses must not be empty")
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors.append(_flatten_gradients(grads, params))
    G = torch.stack(vectors, dim=1)
    gram = G.T @ G
    count = len(losses)
    weights = torch.full((count,), 1.0 / count, dtype=G.dtype, device=G.device)
    spectral = float(torch.linalg.matrix_norm(gram, ord=2).detach().cpu()) if count > 1 else 1.0
    step = min(0.25, 1.0 / max(2.0 * spectral, 1e-6))
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
    """Train CAP-ZW v4 with balanced collision-memory Pareto optimization."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    task_names = ["robustness", "discrimination", "balance", "decorrelation", "entropy_penalty", "diversity"]
    memory_codes = torch.empty((0, config.nbits), dtype=torch.float32, device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)

    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in task_names + ["total", "memory_size"]}
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
            warm_margin = config.margin * (0.70 + 0.30 * progress)
            terms = objective_terms(clean, attacked, labels, warm_margin, memory_codes if progress > 0 else None, memory_labels if progress > 0 else None, config.collision_power, config.topk_negatives, config.diversity_target, config.temperature)
            tasks = [terms[name] for name in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            scale = torch.tensor([config.lambda_robust, config.lambda_collision, config.lambda_balance, config.lambda_corr, config.lambda_entropy, config.lambda_diversity], dtype=mgda.dtype, device=device)
            effective = mgda * scale
            effective = effective / effective.sum().clamp_min(1e-12)
            loss = sum(weight * task for weight, task in zip(effective, tasks))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            memory_codes, memory_labels = _update_memory(memory_codes, memory_labels, clean, labels, config.memory_size)

            for key, value in terms.items():
                totals[key] += float(value.detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            weight_totals += effective.detach().cpu().numpy()
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({key: value / denom for key, value in totals.items()})
        row.update({f"w_{name}": float(weight_totals[i] / denom) for i, name in enumerate(["robust", "discrimination", "balance", "corr", "entropy", "diversity"])})
        history.append(row)

        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "config": config.__dict__, "version": "CAP-ZW-v4"}, path)
    return history


class CAPZWHashNet(HashEncoder):
    """CAP-ZW v4: BEMQ encoder + balanced collision memory + MGDA."""

    def __init__(self, nbits: int = 256, base_channels: int = 32):
        super().__init__(nbits=nbits, base_channels=base_channels, use_bemq=True)
