import torch

from zero_watermarking.v15 import RelationalHashNetV15, V15Config, _binary_collision_terms, _bit_quality


def test_v15_forward_shapes():
    model = RelationalHashNetV15(nbits=32)
    x = torch.rand(4, 1, 128, 128)
    code, emb, rel, logits = model(x, hard=False, return_aux=True)
    assert code.shape == (4, 32)
    assert emb.shape == (4, 128)
    assert rel.shape == (4, 96)
    assert logits.shape == (4, 32)


def test_v15_embeddings_are_normalized():
    model = RelationalHashNetV15(nbits=16)
    x = torch.rand(3, 1, 128, 128)
    _, emb, rel, _ = model(x, hard=False, return_aux=True)
    assert torch.allclose(emb.norm(dim=1), torch.ones(3), atol=1e-4)
    assert torch.allclose(rel.norm(dim=1), torch.ones(3), atol=1e-4)


def test_v15_bit_quality_accepts_balanced_codes():
    codes = torch.cat([torch.zeros(16, 8), torch.ones(16, 8)], dim=0)
    quality, decorrelation, entropy, confidence = _bit_quality(codes, 0.78, 0.15, 0.16)
    assert float(quality) >= 0.0
    assert float(entropy) > 0.99
    assert float(confidence) == 0.0


def test_v15_collision_terms_penalize_near_negatives():
    clean = torch.full((2, 8), 0.5)
    attacked = clean.clone()
    labels = torch.tensor([0, 1])
    tail, mass = _binary_collision_terms(clean, attacked, labels, None, None, 0.30, 2)
    assert float(tail) > 0.0
    assert float(mass) > 0.0
