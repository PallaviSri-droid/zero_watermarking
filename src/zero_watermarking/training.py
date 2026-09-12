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
    lambda_collision: float = 1.0
    lambda_balance: float = 0.10
    lambda_corr: float = 0.05
    lambda_entropy: float = 0.05
    nbits: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 5.0
    memory_size: int = 2048
    mgda_steps: int = 25
    collision_power: float = 2.0


class PairAttackDataset(Dataset):
    """Deterministic clean/attacked pairs for reproducible multi-attack training."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_names: Iterable[str] = ("gaussian_noise", "gaussian_blur", "jpeg", "rotation", "compound"),
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
        return (
            torch.from_numpy(clean[None]),
            torch.from_numpy(np.asarray(attacked, dtype=np.float32)[None]),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            attack_name,
        )


def _pairwise_hamming(z: Tensor) -> Tensor:
    return torch.cdist(z, z, p=1) / z.shape[1]


def objective_terms(
    clean: Tensor,
    attacked: Tensor,
    labels: Tensor,
    margin: float,
    memory_codes: Tensor | None = None,
    memory_labels: Tensor | None = None,
    collision_power: float = 2.0,
) -> Dict[str, Tensor]:
    """CAP-ZW objectives with cross-batch collision-aware hard-negative mining."""
    robustness = (clean - attacked).abs().mean()

    distances = _pairwise_hamming(clean)
    same = labels[:, None].eq(labels[None, :])
    negative = distances.masked_fill(same, float("inf"))
    hard_negative = negative.min(dim=1).values

    if memory_codes is not None and memory_codes.numel() > 0:
        memory_codes = memory_codes.detach()
        memory_labels = memory_labels.detach() if memory_labels is not None else None
        memory_distances = torch.cdist(clean, memory_codes, p=1) / clean.shape[1]
        if memory_labels is not None:
            memory_same = labels[:, None].eq(memory_labels[None, :])
            memory_distances = memory_distances.masked_fill(memory_same, float("inf"))
        memory_hard = memory_distances.min(dim=1).values
        hard_negative = torch.minimum(hard_negative, memory_hard)

    valid = torch.isfinite(hard_negative)
    violation = torch.relu(margin - hard_negative[valid]) if valid.any() else clean.new_zeros(1)
    # Convex hinge plus a stronger penalty for very small/zero inter-image gaps.
    discrimination = violation.mean() if violation.numel() else clean.new_tensor(0.0)
    collision_barrier = (
        (violation.clamp_min(0.0).pow(collision_power)).mean()
        if violation.numel()
        else clean.new_tensor(0.0)
    )
    discrimination = discrimination + 0.5 * collision_barrier

    p = clean.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    balance = (p - 0.5).pow(2).mean()
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()

    centered = clean - clean.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(clean.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    decorrelation = off_diag.pow(2).mean()

    return {
        "robustness": robustness,
        "discrimination": discrimination,
        "balance": balance,
        "decorrelation": decorrelation,
        "entropy_penalty": 1.0 - entropy,
    }


def _flatten_gradients(grads: list[Tensor | None], params: list[Tensor]) -> Tensor:
    chunks = []
    for grad, param in zip(grads, params):
        chunks.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(chunks)


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
    task_count = len(losses)
    weights = torch.full((task_count,), 1.0 / task_count, dtype=G.dtype, device=G.device)
    spectral = float(torch.linalg.matrix_norm(gram, ord=2).detach().cpu()) if task_count > 1 else 1.0
    step = min(0.25, 1.0 / max(2.0 * spectral, 1e-6))
    for _ in range(max(1, steps)):
        weights = _project_simplex(weights - step * (2.0 * gram @ weights))
    return weights.detach()


def _update_memory(
    memory_codes: Tensor,
    memory_labels: Tensor,
    new_codes: Tensor,
    new_labels: Tensor,
    max_size: int,
) -> tuple[Tensor, Tensor]:
    if max_size <= 0:
        return memory_codes[:0], memory_labels[:0]
    codes = torch.cat([memory_codes, new_codes.detach()], dim=0)
    labels = torch.cat([memory_labels, new_labels.detach()], dim=0)
    if codes.shape[0] > max_size:
        codes = codes[-max_size:]
        labels = labels[-max_size:]
    return codes, labels


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    """Train CAP-ZW with cross-batch memory mining and MGDA task balancing."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    task_names = ["robustness", "discrimination", "balance", "decorrelation", "entropy_penalty"]
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
            terms = objective_terms(
                clean,
                attacked,
                labels,
                config.margin,
                memory_codes=memory_codes,
                memory_labels=memory_labels,
                collision_power=config.collision_power,
            )
            tasks = [terms[name] for name in task_names]
            mgda = mgda_weights(tasks, model, steps=config.mgda_steps)
            scale = torch.tensor(
                [config.lambda_robust, config.lambda_collision, config.lambda_balance, config.lambda_corr, config.lambda_entropy],
                dtype=mgda.dtype,
                device=device,
            )
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
        row.update({f"w_{name}": float(weight_totals[i] / denom) for i, name in enumerate(["robust", "discrimination", "balance", "corr", "entropy"])})
        history.append(row)

        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "config": config.__dict__,
                },
                path,
            )
    return history


class CAPZWHashNet(HashEncoder):
    """CAP-ZW v3: BEMQ encoder + cross-batch collision memory + MGDA."""

    def __init__(self, nbits: int = 256, base_channels: int = 32):
        super().__init__(nbits=nbits, base_channels=base_channels, use_bemq=True)
