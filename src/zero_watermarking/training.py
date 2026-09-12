from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
    lambda_balance: float = 0.1
    lambda_corr: float = 0.05
    nbits: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PairAttackDataset(Dataset):
    """Clean/attack image pairs with deterministic per-index attack noise."""
    def __init__(self, images: np.ndarray, labels: np.ndarray, attack_name: str = "gaussian_noise") -> None:
        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)
        if images.ndim != 3:
            raise ValueError("images must have shape [N,H,W]")
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        if attack_name not in ATTACKS or attack_name == "clean":
            raise ValueError("attack_name must be a non-clean attack")
        self.images = np.clip(images, 0.0, 1.0)
        self.labels = labels.astype(np.int64)
        self.attack_name = attack_name

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        clean = self.images[index]
        kwargs = {"sigma": 0.03, "seed": index} if self.attack_name == "gaussian_noise" else {"seed": index} if self.attack_name == "compound" else {}
        attacked = ATTACKS[self.attack_name](clean, **kwargs)
        return torch.from_numpy(clean[None]), torch.from_numpy(attacked[None]), torch.tensor(int(self.labels[index]), dtype=torch.long)


def _pairwise(z: torch.Tensor) -> torch.Tensor:
    return torch.cdist(z, z, p=1) / z.shape[1]


def objective_terms(clean: torch.Tensor, attacked: torch.Tensor, labels: torch.Tensor, margin: float) -> Dict[str, torch.Tensor]:
    robust = (clean - attacked).abs().mean()
    balance = (clean.mean(0) - 0.5).pow(2).mean()
    centered = clean - clean.mean(0, keepdim=True)
    cov = centered.T @ centered / max(clean.shape[0] - 1, 1)
    corr = (cov - torch.diag(torch.diagonal(cov))).pow(2).mean()
    distances = _pairwise(clean)
    same = labels[:, None].eq(labels[None, :])
    # Exclude self-distances and same-label distances from the hard-negative set.
    negative = distances.masked_fill(same, float("inf"))
    hard = negative.min(1).values
    collision = torch.relu(margin - hard).mean()
    return {"robust": robust, "collision": collision, "balance": balance, "corr": corr}


def mgda_weights(losses: list[torch.Tensor], model: nn.Module) -> torch.Tensor:
    """Normalized inverse-gradient task weighting as a lightweight MGDA approximation."""
    params = [p for p in model.parameters() if p.requires_grad]
    norms = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squared = sum(g.detach().pow(2).sum() for g in grads if g is not None)
        norms.append(torch.sqrt(squared + 1e-12))
    inverse = torch.stack([1.0 / value for value in norms])
    return inverse / inverse.sum().clamp_min(1e-12)


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = {k: 0.0 for k in ["robust", "collision", "balance", "corr", "total"]}
        for clean_x, attacked_x, labels in loader:
            clean_x, attacked_x, labels = clean_x.to(device), attacked_x.to(device), labels.to(device)
            clean = torch.sigmoid(model(clean_x, hard=False))
            attacked = torch.sigmoid(model(attacked_x, hard=False))
            terms = objective_terms(clean, attacked, labels, config.margin)
            tasks = [terms["robust"], terms["collision"], terms["balance"], terms["corr"]]
            weights = mgda_weights(tasks, model)
            loss = (weights[0] * config.lambda_robust * tasks[0]
                    + weights[1] * config.lambda_collision * tasks[1]
                    + weights[2] * config.lambda_balance * tasks[2]
                    + weights[3] * config.lambda_corr * tasks[3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in terms.items(): total[key] += float(value.detach())
            total["total"] += float(loss.detach())
        n = max(len(loader), 1)
        history.append({"epoch": epoch + 1, **{k: v / n for k, v in total.items()}})
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}, path)
    return history


class CAPZWHashNet(HashEncoder):
    """Named model used by experiments and notebooks."""
    def __init__(self, nbits: int = 256):
        super().__init__(nbits=nbits)
