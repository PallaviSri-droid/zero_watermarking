import numpy as np
import torch

from zero_watermarking.training import CAP_ZW_VERSION, PairAttackDataset, objective_terms


def test_version_is_single_source_of_truth():
    assert CAP_ZW_VERSION == "CAP-ZW-v8"


def test_multiview_dataset_length_and_labels():
    images = np.zeros((4, 16, 16), dtype=np.float32)
    labels = np.arange(4, dtype=np.int64)
    dataset = PairAttackDataset(images, labels, attack_views=3)
    assert len(dataset) == 12
    _, _, first_label, _ = dataset[0]
    _, _, second_view_label, _ = dataset[1]
    assert int(first_label) == 0
    assert int(second_view_label) == 0


def test_objective_returns_discrete_collision_term():
    clean = torch.tensor(
        [[0.9, 0.1, 0.9, 0.1], [0.9, 0.1, 0.9, 0.1], [0.1, 0.9, 0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    attacked = clean.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    terms = objective_terms(
        clean,
        attacked,
        labels,
        margin=0.30,
        binary_collision_target=0.50,
        topk_negatives=2,
    )
    assert "binary_collision" in terms
    assert torch.isfinite(terms["binary_collision"])
    total = sum(terms.values())
    total.backward()
    assert clean.grad is not None
