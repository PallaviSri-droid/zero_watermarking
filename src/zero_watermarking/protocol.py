from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import torch

from .attacks import ATTACKS


@dataclass(frozen=True)
class AttackSpec:
    name: str
    parameter: str
    values: tuple[float, ...]


# Locked main-paper attack grid. Every reproduced method must use these exact
# attacks/strengths unless a result is explicitly labeled exploratory.
DEFAULT_ATTACK_GRID: tuple[AttackSpec, ...] = (
    AttackSpec("gaussian_noise", "sigma", (0.03, 0.08)),
    AttackSpec("gaussian_blur", "sigma", (1.0, 2.0)),
    AttackSpec("jpeg", "quality", (70.0, 40.0)),
    AttackSpec("rotation", "degrees", (5.0, 10.0)),
    AttackSpec("crop_resize", "fraction", (0.05,)),
    AttackSpec("translation", "pixels", (3.0,)),
    AttackSpec("compound", "seed", (23.0,)),
)


def attack_grid(
    image: np.ndarray,
    specs: tuple[AttackSpec, ...] = DEFAULT_ATTACK_GRID,
    seed: int = 42,
) -> dict[str, dict[float, np.ndarray]]:
    """Generate the locked attack-strength grid for an identical benchmark protocol."""
    outputs: dict[str, dict[float, np.ndarray]] = {}
    for spec in specs:
        outputs[spec.name] = {}
        for value in spec.values:
            kwargs: dict[str, float | int] = {spec.parameter: value}
            if spec.name == "compound":
                kwargs["seed"] = int(value) + int(seed) % 100000
            elif spec.name in {"gaussian_noise"}:
                kwargs["seed"] = int(seed)
            outputs[spec.name][value] = ATTACKS[spec.name](image, **kwargs)
    return outputs


def paired_labels(n_images: int) -> tuple[np.ndarray, np.ndarray]:
    """Return labels for positive (same-image) and negative (different-image) pairs."""
    positives = np.ones(n_images, dtype=np.int8)
    negatives = np.zeros(n_images * (n_images - 1) // 2, dtype=np.int8)
    return positives, negatives


def all_negative_pairs(ids: list[int]) -> list[tuple[int, int]]:
    """Enumerate unique different-image pairs for collision/discrimination tests."""
    return list(combinations(ids, 2))


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments.

    Deterministic kernels are requested where supported. If an operation has no
    deterministic implementation on a particular backend, PyTorch may still
    raise an error at execution time; that is preferable to silently claiming
    reproducibility.
    """
    import os
    import random

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
