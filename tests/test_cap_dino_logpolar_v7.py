from __future__ import annotations

import torch

from zero_watermarking.cap_dino_logpolar_v7 import V7Config, _hard_robustness_terms


def test_hard_robustness_tail_keeps_gradient() -> None:
    cfg = V7Config()
    clean = torch.rand(8, 32, requires_grad=True)
    attacked = (clean + 0.15 * torch.randn_like(clean)).clamp(0.0, 1.0)
    mean_loss, tail_loss, _ = _hard_robustness_terms(clean, attacked, cfg)
    (mean_loss + tail_loss).backward()
    assert clean.grad is not None
    assert torch.isfinite(clean.grad).all()
    assert float(clean.grad.abs().sum()) > 0.0


def test_v7_gate_limits_are_valid() -> None:
    cfg = V7Config()
    assert 0.0 < cfg.gate_min_usage < cfg.gate_max_usage < 1.0
    assert 0.0 < cfg.gate_row_max_target < 1.0
    assert 0.0 < cfg.branch_dropout < 0.5
