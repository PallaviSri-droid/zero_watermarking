import torch

from zero_watermarking.v14 import _hard_entropy_and_balance, _robustness_cap


def test_v14_hard_entropy_prefers_balanced_bits():
    collapsed = torch.full((32, 16), 0.05)
    balanced = torch.cat([torch.zeros(16, 16), torch.ones(16, 16)], dim=0)
    c_loss, _, _, _ = _hard_entropy_and_balance(collapsed, 0.82, 0.12)
    b_loss, _, _, _ = _hard_entropy_and_balance(balanced, 0.82, 0.12)
    assert c_loss > b_loss


def test_v14_robustness_cap_penalizes_large_worst_case():
    clean = torch.zeros(4, 8)
    attacked = clean.clone()
    attacked[-1] = 1.0
    assert _robustness_cap(clean, attacked, 0.10) > 0


def test_v14_balanced_code_has_low_balance_penalty():
    balanced = torch.cat([torch.zeros(16, 8), torch.ones(16, 8)], dim=0)
    _, balance_loss, _, observed_balance = _hard_entropy_and_balance(balanced, 0.82, 0.12)
    assert float(balance_loss) == 0.0
    assert float(observed_balance) == 0.0
