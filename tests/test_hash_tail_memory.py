import torch

from zero_watermarking.hash_tail_memory import memory_hard_negative_loss


def test_memory_hard_negative_backpropagates_only_to_queries() -> None:
    queries = torch.tensor(
        [[0.50, 0.50, 0.50, 0.50], [0.10, 0.90, 0.10, 0.90]],
        dtype=torch.float32,
        requires_grad=True,
    )
    memory = torch.tensor(
        [[0.50, 0.50, 0.50, 0.50], [1.00, 0.00, 1.00, 0.00], [0.10, 0.90, 0.10, 0.90]],
        dtype=torch.float32,
        requires_grad=True,
    )
    query_labels = torch.tensor([10, 20])
    memory_labels = torch.tensor([99, 10, 20])

    loss = memory_hard_negative_loss(
        queries,
        query_labels,
        memory,
        memory_labels,
        margin=0.32,
        temperature=0.08,
        topk=2,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert memory.grad is None


def test_memory_hard_negative_is_zero_without_cross_image_memory() -> None:
    queries = torch.full((2, 4), 0.5, requires_grad=True)
    memory = torch.full((2, 4), 0.0)
    labels = torch.tensor([1, 2])
    memory_labels = torch.tensor([1, 2])

    loss = memory_hard_negative_loss(queries, labels, memory, memory_labels)

    assert float(loss) == 0.0
