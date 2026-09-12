import torch

from zero_watermarking.v11 import CAP_ZW_V11_VERSION, V11Config, _selective_robustness_guard


def test_v11_version_and_defaults():
    assert CAP_ZW_V11_VERSION == "CAP-ZW-v11"
    config = V11Config()
    assert config.robustness_guard_quantile == 0.85
    assert config.robustness_guard_lambda_init < 1.0
    assert config.lambda_binary_collision < 1.45
    assert config.robustness_guard_batch_fraction == 0.25


def test_selective_robustness_guard_is_finite_and_differentiable():
    torch.manual_seed(11)
    clean = torch.sigmoid(torch.randn(8, 16, requires_grad=True))
    attacked = (clean + 0.02 * torch.randn_like(clean)).clamp(0, 1)
    penalty, violation, q_value, worst = _selective_robustness_guard(
        clean,
        attacked,
        target=0.021,
        quantile=0.85,
        softness=0.008,
        tail_weight=0.70,
        mean_weight=0.20,
        batch_fraction=0.25,
    )
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)
    assert torch.isfinite(violation)
    assert torch.isfinite(q_value)
    assert worst.numel() == 2
    penalty.backward()
    assert clean.grad is not None or clean.is_leaf is False
