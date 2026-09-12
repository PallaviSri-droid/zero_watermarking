from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn

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
    hard = distances.masked_fill(same, float("inf")).min(1).values
    collision = torch.relu(margin - hard).mean()
    return {"robust": robust, "collision": collision, "balance": balance, "corr": corr}


def mgda_weights(losses: list[torch.Tensor], model: nn.Module) -> torch.Tensor:
    """Normalized inverse-gradient weighting; practical MGDA approximation."""
    params = [p for p in model.parameters() if p.requires_grad]
    norms = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squared = sum(g.detach().pow(2).sum() for g in grads if g is not None)
        norms.append(torch.sqrt(squared + 1e-12))
    inv = torch.stack([1.0 / n for n in norms])
    return inv / inv.sum().clamp_min(1e-12)


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
            loss = (
                weights[0] * config.lambda_robust * tasks[0]
                + weights[1] * config.lambda_collision * tasks[1]
                + weights[2] * config.lambda_balance * tasks[2]
                + weights[3] * config.lambda_corr * tasks[3]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in terms.items():
                total[key] += float(value.detach())
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
