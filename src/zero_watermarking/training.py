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
    """Deterministic clean/attacked pairs for reproducible training."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_names: Iterable[str] = ("gaussian_noise", "blur", "jpeg", "rotation", "compound"),
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
        if name == "gaussian_noise":
            return ATTACKS[name](clean, sigma=0.03, seed=index)
        if name == "compound":
            return ATTACKS[name](clean, seed=index)
        # Deterministic fixed-strength defaults for the training augmentation grid.
        defaults = {
            "blur": {"sigma": 1.0},
            "median_blur": {"kernel": 3},
            "salt_pepper": {"amount": 0.01, "seed": index},
            "jpeg": {"quality": 70},
            "brightness": {"factor": 1.10},
            "contrast": {"factor": 1.15},
            "rotation": {"angle": 5.0},
            "crop_resize": {"fraction": 0.90},
            "translation": {"dx": 3, "dy": 2},
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
    if valid.any():
        discrimination = torch.relu(margin - hard_negative[valid]).mean()
    else:
        discrimination = clean.new_tensor(0.0)

    p = clean.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    balance = (p - 0.5).pow(2).mean()
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()

    centered = clean - clean.mean(dim=0, keepdim=True)
    denom = max(clean.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
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
    """Euclidean projection onto {w >= 0, sum(w)=1}."""
    if x.numel() == 1:
        return torch.ones_like(x)
    u, _ = torch.sort(x, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    cond = u - cssv / idx > 0
    rho = torch.where(cond)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(x - theta, min=0.0)


def mgda_weights(
    losses: list[Tensor],
    model: nn.Module,
    steps: int = 25,
    lr: float = 0.25,
) -> Tensor:
    """Approximate the MGDA minimum-norm solution on the task simplex.

    We form one gradient vector per objective and solve
        min_w ||sum_i w_i g_i||^2  subject to w_i >= 0, sum_i w_i = 1
    with projected gradient descent. The weights are detached because they are
    determined by the current task gradients rather than optimized parameters.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not losses:
        raise ValueError("losses must not be empty")
    gradient_vectors = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        gradient_vectors.append(_flatten_gradients(grads, params))
    G = torch.stack(gradient_vectors, dim=1)  # [P, T]
    gram = G.T @ G
    n_tasks = len(losses)
    weights = torch.full((n_tasks,), 1.0 / n_tasks, device=G.device, dtype=G.dtype)
    lipschitz = float(2.0 * torch.linalg.matrix_norm(gram, ord=2).detach().cpu()) if n_tasks > 1 else 1.0
    step = min(lr, 1.0 / max(lipschitz, 1e-6))
    for _ in range(max(1, steps)):
        grad_w = 2.0 * (gram @ weights)
        weights = _project_simplex(weights - step * grad_w)
    return weights.detach()


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    """Train CAP-ZW with MGDA-balanced robustness/discrimination/hash-quality tasks."""
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        totals = {k: 0.0 for k in ["robustness", "discrimination", "balance", "decorrelation", "entropy_penalty", "total"]}
        weight_totals = np.zeros(5, dtype=np.float64)
        batches = 0

        for batch in loader:
            if len(batch) == 4:
                clean_x, attacked_x, labels, _ = batch
            else:
                clean_x, attacked_x, labels = batch
            clean_x = clean_x.to(device, non_blocking=True)
            attacked_x = attacked_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # BEMQ hard=False already returns probabilities; do NOT sigmoid twice.
            clean = model(clean_x, hard=False)
            attacked = model(attacked_x, hard=False)
            terms = objective_terms(clean, attacked, labels, config.margin)
            tasks = [
                terms["robustness"],
                terms["discrimination"],
                terms["balance"],
                terms["decorrelation"],
                terms["entropy_penalty"],
            ]
            mgda = mgda_weights(tasks, model)
            scaled = torch.tensor(
                [
                    config.lambda_robust,
                    config.lambda_collision,
                    config.lambda_balance,
                    config.lambda_corr,
                    config.lambda_entropy,
                ],
                device=device,
                dtype=mgda.dtype,
            )
            effective = mgda * scaled
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
        for index, name in enumerate(["w_robust", "w_discrimination", "w_balance", "w_corr", "w_entropy"]):
            row[name] = float(weight_totals[index] / denom)
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
