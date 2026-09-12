from __future__ import annotations

"""Locked CAP-ZW research candidate.

This module deliberately does not create another hand-tuned model version.  It
combines the empirically useful v9 collision machinery with the v11 selective
robustness guard and exposes one configuration for controlled multi-seed
selection.  The checkpoint is *not* considered final until the locked protocol
selects it from multiple seeds.
"""

from dataclasses import dataclass

from .v11 import (
    CAPZWHashNet,
    PairAttackDataset,
    V11Config,
    _selective_robustness_guard,
    train_cap_zw_v11,
)


CAP_ZW_FINAL = "CAP-ZW-final-candidate"


@dataclass
class FinalCAPZWConfig(V11Config):
    """Pre-registered starting point for the final CAP-ZW candidate family.

    Defaults intentionally sit between v9 and v11 rather than copying the
    strongest collision penalty.  This is a hypothesis to be tested, not a
    claim of superiority.
    """

    # Preserve v11's selective robustness protection.
    robustness_guard_target: float = 0.021
    robustness_guard_quantile: float = 0.85
    robustness_guard_softness: float = 0.008
    robustness_guard_lambda_init: float = 0.55
    robustness_guard_lambda_growth: float = 0.10
    robustness_guard_lambda_max: float = 2.50
    robustness_guard_tail_weight: float = 0.70
    robustness_guard_mean_weight: float = 0.20
    robustness_guard_batch_fraction: float = 0.25

    # Moderate collision pressure: below v9's 1.45 while retaining v11's
    # hard-negative/mass-aware objective.
    lambda_binary_collision: float = 1.32
    lambda_consistency: float = 1.00
    lambda_corr: float = 0.25
    lambda_diversity: float = 0.30
    topk_negatives: int = 12
    memory_size: int = 4096
    memory_warmup: int = 384
    collision_power: float = 2.0
    binary_collision_target: float = 0.135
    tail_target: float = 0.26
    diversity_target: float = 0.36


def build_final_config(**overrides: object) -> FinalCAPZWConfig:
    """Build the locked candidate configuration with explicit overrides."""
    return FinalCAPZWConfig(**overrides)


__all__ = [
    "CAP_ZW_FINAL",
    "FinalCAPZWConfig",
    "build_final_config",
    "CAPZWHashNet",
    "PairAttackDataset",
    "train_cap_zw_v11",
    "_selective_robustness_guard",
]
