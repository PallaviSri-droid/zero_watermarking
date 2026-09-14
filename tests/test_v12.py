import torch

from zero_watermarking.v12 import (
    _binary_confidence_loss,
    _entropy_floor_loss,
    _separation_tail_loss,
)


def test_v12_separation_tail_penalizes_low_tail():
    distances = torch.tensor([[0.02, 0.04, 0.12, 0.30], [0.03, 0.05, 0.20, 0.40]])
    loss = _separation_tail_loss(distances, 0.10, 0.15, torch.device("cpu"))
    assert float(loss) > 0.0


def test_v12_entropy_floor_penalizes_collapsed_bits():
    collapsed = torch.full((32, 16), 0.02)
    balanced = torch.cat([torch.zeros(16, 16), torch.ones(16, 16)], dim=0)
    assert _entropy_floor_loss(collapsed, 0.82) > _entropy_floor_loss(balanced, 0.82)


def test_v12_binary_confidence_prefers_decisive_bits():
    ambiguous = torch.full((8, 16), 0.5)
    decisive = torch.cat([torch.zeros(4, 16), torch.ones(4, 16)], dim=0)
    assert _binary_confidence_loss(ambiguous, 0.18) > _binary_confidence_loss(decisive, 0.18)
