import torch

from zero_watermarking.training import CAP_ZW_VERSION, objective_terms


def test_objective_terms_current_version_shape_and_finite():
    torch.manual_seed(7)
    clean = torch.sigmoid(torch.randn(6, 16))
    attacked = (clean + 0.02 * torch.randn_like(clean)).clamp(0, 1)
    labels = torch.arange(6)
    memory = torch.sigmoid(torch.randn(12, 16))
    memory_labels = torch.arange(12) + 100
    terms = objective_terms(
        clean,
        attacked,
        labels,
        margin=0.42,
        memory_codes=memory,
        memory_labels=memory_labels,
        topk_negatives=4,
        diversity_target=0.45,
        tail_target=0.34,
        uniformity_temperature=0.08,
    )
    expected = {
        "robustness",
        "tail_collision",
        "diversity",
        "balance",
        "decorrelation",
        "entropy_penalty",
        "uniformity",
        "consistency",
        "binary_collision",
    }
    assert CAP_ZW_VERSION == "CAP-ZW-v8"
    assert set(terms) == expected
    assert all(torch.isfinite(value).item() for value in terms.values())
    assert all(value.ndim == 0 for value in terms.values())


def test_objective_terms_rejects_image_tensor_shape():
    clean = torch.rand(2, 1, 32, 32)
    attacked = torch.rand(2, 1, 32, 32)
    labels = torch.arange(2)
    try:
        objective_terms(clean, attacked, labels, margin=0.4)
    except ValueError as exc:
        assert "[B, nbits]" in str(exc)
    else:
        raise AssertionError("Expected objective_terms to reject non-hash tensors")
