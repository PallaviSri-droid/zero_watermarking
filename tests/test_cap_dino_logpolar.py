import torch

from zero_watermarking.cap_dino_logpolar import AdaptiveTriBranchGate, FusionConfig, fusion_loss, fusion_objective


def test_gate_is_per_sample_and_sums_to_one():
    gate = AdaptiveTriBranchGate(32)
    a = torch.randn(5, 32)
    b = torch.randn(5, 32)
    c = torch.randn(5, 32)
    weights = gate(a, b, c)
    assert weights.shape == (5, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.all(weights >= 0)


def test_fusion_objective_returns_finite_terms():
    clean = torch.sigmoid(torch.randn(4, 16))
    attacked = torch.sigmoid(torch.randn(4, 16))
    labels = torch.arange(4)
    gates = torch.softmax(torch.randn(8, 3), dim=1)
    terms = fusion_objective(clean, attacked, labels, gates)
    loss = fusion_loss(terms)
    assert torch.isfinite(loss)
    assert 0.0 <= float(terms["bit_entropy"]) <= 1.0


def test_fusion_config_defaults():
    cfg = FusionConfig()
    assert cfg.bits == 128
    assert cfg.cap_weight > 0
    assert cfg.dino_weight > 0
    assert cfg.logpolar_weight > 0
