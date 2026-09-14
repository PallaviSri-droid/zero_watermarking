from __future__ import annotations

"""Memory-backed hard-negative supervision for the hash tail.

The loss keeps the memory entries detached and backpropagates only through the
current query codes. It is deliberately small and is intended as a controlled
repair for sparse mini-batch negative coverage rather than a new architecture.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def memory_hard_negative_loss(
    queries: Tensor,
    labels: Tensor,
    memory_codes: Tensor | None,
    memory_labels: Tensor | None,
    *,
    margin: float = 0.32,
    temperature: float = 0.08,
    topk: int = 24,
) -> Tensor:
    """Push current codes away from the closest stored codes of other images.

    ``queries`` may contain soft Bernoulli codes in [0, 1], while the memory is
    expected to contain detached binary codes. The normalized L1 distance is the
    expected Hamming distance for Bernoulli queries against binary memory codes.
    """
    if memory_codes is None or memory_codes.numel() == 0:
        return queries.new_zeros(())
    if memory_labels is None or memory_labels.numel() == 0:
        return queries.new_zeros(())
    if queries.ndim != 2 or memory_codes.ndim != 2:
        raise ValueError("queries and memory_codes must be 2D tensors")
    if queries.shape[1] != memory_codes.shape[1]:
        raise ValueError("queries and memory_codes must have equal bit dimensions")

    memory = memory_codes.detach().to(device=queries.device, dtype=queries.dtype)
    labels_q = labels.to(device=queries.device)
    labels_m = memory_labels.detach().to(device=queries.device)

    distances = torch.cdist(queries.clamp(0.0, 1.0), memory, p=1) / queries.shape[1]
    same = labels_q[:, None].eq(labels_m[None, :])
    distances = distances.masked_fill(same, float("inf"))
    finite = torch.isfinite(distances)
    if not finite.any():
        return queries.new_zeros(())

    fill = torch.where(finite, distances, torch.full_like(distances, 2.0))
    valid_per_row = finite.sum(dim=1)
    valid_rows = valid_per_row > 0
    if not valid_rows.any():
        return queries.new_zeros(())

    kk = min(max(int(topk), 1), int(memory.shape[0]))
    nearest = torch.topk(fill[valid_rows], kk, largest=False, dim=1).values
    tau = max(float(temperature), 1e-4)
    pair_loss = F.softplus((float(margin) - nearest) / tau).mul(tau)
    return pair_loss.mean()


__all__ = ["memory_hard_negative_loss"]
