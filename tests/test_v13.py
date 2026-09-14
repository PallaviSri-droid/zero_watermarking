import torch

from zero_watermarking.v13 import V13Config
from zero_watermarking.v12 import _binary_confidence_loss, _entropy_floor_loss, _separation_tail_loss


def test_v13_config_is_collision_first_but_robustness_bounded():
    cfg = V13Config()
    assert cfg.lambda_separation_constraint > 0
    assert cfg.lambda_entropy_constraint > 0
    assert cfg.guard_target_start > cfg.guard_target_final


def test_v13_separation_tail_penalizes_low_tail():
    d = torch.tensor([[0.01, 0.03, 0.12], [0.02, 0.05, 0.20]])
    assert float(_separation_tail_loss(d, 0.11, 0.16, torch.device("cpu"))) > 0


def test_v13_anti_collapse_terms_behave_directionally():
    collapsed = torch.full((32, 16), 0.02)
    balanced = torch.cat([torch.zeros(16, 16), torch.ones(16, 16)], dim=0)
    assert _entropy_floor_loss(collapsed, 0.80) > _entropy_floor_loss(balanced, 0.80)

    ambiguous = torch.full((8, 16), 0.5)
    decisive = torch.cat([torch.zeros(4, 16), torch.ones(4, 16)], dim=0)
    assert _binary_confidence_loss(ambiguous, 0.16) > _binary_confidence_loss(decisive, 0.16)
