import torch

from src.training.losses import captioning_loss


def _logits_from_logprobs(prob_matrix):
    b, t, v = prob_matrix.shape
    logits = torch.log(torch.clamp(prob_matrix, min=1e-12))
    return logits


def test_shift_alignment():
    v = 6
    logits = torch.zeros(2, 3, v)
    logits[:, 0, 4] = 100.0
    logits[:, 1, 5] = 100.0
    logits[:, 2, 2] = 100.0  # position 2 is consumed but predicts nothing
    labels = torch.tensor([[1, 4, 5], [1, 4, 5]])  # <SOS>, token4, token5
    loss = captioning_loss(logits, labels, pad_id=0)
    assert loss.item() == 0.0


def test_loss_matches_manual_ce():
    torch.manual_seed(0)
    v = 8
    logits = torch.randn(3, 5, v)
    labels = torch.randint(1, v, (3, 5))
    labels[:, 0] = 1
    labels[:, -1] = 2
    loss = captioning_loss(logits, labels, pad_id=0)

    shifted_logits = logits[:, :-1].reshape(-1, v)
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels != 0
    expected = torch.nn.functional.cross_entropy(
        shifted_logits[valid], shifted_labels[valid]
    )
    assert torch.allclose(loss, expected)


def test_pad_positions_have_no_loss():
    v = 6
    logits = torch.zeros(2, 4, v)
    logits[:, 0, 4] = 100.0
    logits[:, 1, 5] = 100.0
    labels = torch.tensor([[1, 4, 5, 0], [1, 4, 5, 0]])
    loss = captioning_loss(logits, labels, pad_id=0)
    assert loss.item() == 0.0


def test_uniform_logits_loss_is_log_v():
    v = 7
    logits = torch.zeros(3, 4, v)
    labels = torch.randint(1, v, (3, 4))
    labels[:, 0] = 1
    loss = captioning_loss(logits, labels, pad_id=0)
    from math import log as mlog

    assert torch.isclose(loss, torch.tensor(mlog(v)).float(), atol=1e-4)


def test_empty_no_target_returns_zero():
    v = 6
    logits = torch.zeros(2, 1, v)
    labels = torch.zeros(2, 1, dtype=torch.long)
    assert captioning_loss(logits, labels, pad_id=0).item() == 0.0


def test_grad_flows_through_loss():
    v = 6
    logits = torch.randn(2, 4, v, requires_grad=True)
    labels = torch.tensor([[1, 3, 2, 0], [1, 4, 5, 0]])
    loss = captioning_loss(logits, labels, pad_id=0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()