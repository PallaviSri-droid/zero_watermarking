from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .learned import HashEncoder, STEBinary


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    margin: float = 0.30
    lambda_robust: float = 1.0
    lambda_collision: float = 1.0
    lambda_balance: float = 0.1
    lambda_corr: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _pairwise(z):
    return torch.cdist(z, z, p=1) / z.shape[1]


def objective_terms(clean, attacked, labels, margin):
    robust = (clean - attacked).abs().mean()
    balance = (clean.mean(0) - 0.5).pow(2).mean()
    centered = clean - clean.mean(0, keepdim=True)
    cov = centered.T @ centered / max(clean.shape[0] - 1, 1)
    corr = (cov - torch.diag(torch.diagonal(cov))).pow(2).mean()
    d = _pairwise(clean)
    same = labels[:, None].eq(labels[None, :])
    hard = d.masked_fill(same, float("inf")).min(1).values
    collision = torch.relu(margin - hard).mean()
    return {"robust": robust, "collision": collision, "balance": balance, "corr": corr}


def mgda_weights(losses, model):
    """Simple normalized-gradient MGDA approximation for scalar task losses."""
    grads = []
    params = [p for p in model.parameters() if p.requires_grad]
    for loss in losses:
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        norm = torch.sqrt(sum((x.detach().pow(2).sum() for x in g if x is not None)) + 1e-12)
        grads.append(norm)
    inv = torch.stack([1.0 / g for g in grads])
    return inv / inv.sum()


def train_cap_zw(model: nn.Module, loader, config: TrainConfig, checkpoint: str | None = None):
    device = torch.device(config.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history = []
    for epoch in range(config.epochs):
        model.train(); total = {k: 0.0 for k in ["robust", "collision", "balance", "corr", "total"]}
        for clean_x, attacked_x, labels in loader:
            clean_x, attacked_x, labels = clean_x.to(device), attacked_x.to(device), labels.to(device)
            clean_logits = model(clean_x, hard=False)
            attacked_logits = model(attacked_x, hard=False)
            clean = torch.sigmoid(clean_logits)
            attacked = torch.sigmoid(attacked_logits)
            terms = objective_terms(clean, attacked, labels, config.margin)
            tasks = [terms["robust"], terms["collision"], terms["balance"], terms["corr"]]
            w = mgda_weights(tasks, model)
            loss = (w[0] * config.lambda_robust * tasks[0] +
                    w[1] * config.lambda_collision * tasks[1] +
                    w[2] * config.lambda_balance * tasks[2] +
                    w[3] * config.lambda_corr * tasks[3])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            for k, v in terms.items(): total[k] += float(v.detach())
            total["total"] += float(loss.detach())
        n = max(len(loader), 1)
        row = {"epoch": epoch + 1, **{k: v / n for k, v in total.items()}}
        history.append(row)
        if checkpoint:
            p = Path(checkpoint); p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": opt.state_dict(), "history": history}, p)
    return history


class CAPZWHashNet(HashEncoder):
    """Named model used by the paper experiments."""
    pass
