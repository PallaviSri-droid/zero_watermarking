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


def objective_terms(clean: Tensor, attacked: Tensor, labels: Tensor, margin: float) -> Dict[str, Tensor]:
    """CAP-ZW objectives in normalized hash space [0,1]."""
    robustness = (clean - attacked).abs().mean()

    distances = _pairwise_hamming(clean)
    same = labels[:, None].eq(labels[None, :])
    negative = distances.masked_fill(same, float("inf"))
    hard_negative = negative.min(dim=1).values
    valid = torch.isfinite(hard_negative)
    discrimination = (
        torch.relu(margin - hard_negative[valid]).mean()
        if valid.any()
        else clean.new_tensor(0.0)
    )

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
    """Euclidean projection onto the probability simplex."""
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
    """Compute a minimum-norm multi-gradient simplex weighting.

    The task weights solve min ||sum_i w_i g_i||^2 subject to w >= 0 and sum(w)=1.
    Projected gradient descent on the task Gram matrix gives a deterministic,
    lightweight MGDA-style solver without adding an external dependency.
    """
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


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    """Train CAP-ZW using collision-aware objectives and MGDA task balancing."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    task_names = ["robustness", "discrimination", "balance", "decorrelation", "entropy_penalty"]
    totals_keys = task_names + ["total"]

    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in totals_keys}
        weight_totals = np.zeros(len(task_names), dtype=np.float64)
        batches = 0

        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x = clean_x.to(device, non_blocking=True)
            attacked_x = attacked_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # BEMQ(hard=False) already returns probabilities; avoid a second sigmoid.
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)
            terms = objective_terms(clean, attacked, labels, config.margin)
            tasks = [terms[name] for name in task_names]
            mgda = mgda_weights(tasks, model)
            scale = torch.tensor(
                [
                    config.lambda_robust,
                    config.lambda_collision,
                    config.lambda_balance,
                    config.lambda_corr,
                    config.lambda_entropy,
                ],
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

            for key, value in terms.items():
                totals[key] += float(value.detach())
            totals["total"] += float(loss.detach())
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
    """Publication experiment model: BEMQ + collision-aware Pareto objectives."""

    def __init__(self, nbits: int = 256, base_channels: int = 32):
        super().__init__(nbits=nbits, base_channels=base_channels, use_bemq=True)
