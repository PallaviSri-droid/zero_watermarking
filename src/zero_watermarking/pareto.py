from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class MGDAResult:
    weights: Tensor
    objective_values: Tensor


def _flatten_gradients(loss: Tensor, parameters: Sequence[Tensor]) -> Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    chunks = []
    for param, grad in zip(parameters, grads):
        chunks.append((torch.zeros_like(param) if grad is None else grad).reshape(-1))
    return torch.cat(chunks)


def simplex_project(v: Tensor) -> Tensor:
    """Project a vector onto the probability simplex."""
    if v.ndim != 1:
        raise ValueError("v must be one-dimensional")
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    rho = torch.where(u - cssv / idx > 0)[0]
    if len(rho) == 0:
        return torch.full_like(v, 1.0 / n)
    r = rho[-1]
    theta = cssv[r] / (r + 1).to(v.dtype)
    return torch.clamp(v - theta, min=0)


def approximate_mgda_weights(losses: Sequence[Tensor], parameters: Sequence[Tensor]) -> MGDAResult:
    """Small, transparent MGDA-style solver for research prototypes.

    It greedily chooses a convex combination of task gradients by solving a
    simplex-constrained quadratic problem with projected gradient descent.
    """
    if len(losses) < 2:
        return MGDAResult(torch.ones(1, device=losses[0].device), torch.stack(list(losses)))

    gradients = torch.stack([_flatten_gradients(loss, parameters) for loss in losses])
    gram = gradients @ gradients.T
    weights = torch.full((len(losses),), 1.0 / len(losses), device=gram.device)
    step = 0.2
    for _ in range(80):
        grad = 2 * gram @ weights
        weights = simplex_project(weights - step * grad)
        step *= 0.99
    return MGDAResult(weights.detach(), torch.stack(list(losses)).detach())
