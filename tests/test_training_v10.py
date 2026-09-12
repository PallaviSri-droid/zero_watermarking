import torch

from zero_watermarking.v10 import CAP_ZW_V10_VERSION, V10Config, _robustness_guard


def test_v10_version_and_defaults():
    assert CAP_ZW_V10_VERSION == "CAP-ZW-v10"
    config = V10Config()
    assert config.robustness_guard_target < config.robust_target
    assert config.lambda_binary_collision < 1.45
    assert config.robustness_guard_quantile == 0.90


def test_robustness_guard_is_finite_and_differentiable():
    clean = torch.sigmoid(torch.randn(8, 16))
    clean.retain_grad()
    attacked = (clean + 0.02 * torch.randn_like(clean)).clamp(0, 1)
    penalty, violation, q_value = _robustness_guard(
        clean,
        attacked,
        target=0.019,
        quantile=0.90,
        softness=0.006,
        mean_weight=0.50,
    )
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)
    assert torch.isfinite(violation)
    assert torch.isfinite(q_value)
    penalty.backward()
    assert clean.grad is not None
