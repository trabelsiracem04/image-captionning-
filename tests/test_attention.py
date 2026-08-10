import torch

from src.models.attention import Attention


def _make_attention(**kwargs):
    enc = 2048
    dec = 512
    att = 512
    return Attention(
        encoder_dim=kwargs.get("encoder_dim", enc),
        decoder_dim=kwargs.get("decoder_dim", dec),
        attention_dim=kwargs.get("attention_dim", att),
    )


def _make_inputs(b=2, n=49, enc=2048, dec=512):
    features = torch.randn(b, n, enc)
    hidden = torch.randn(b, dec)
    return features, hidden


def test_forward_shapes():
    attn = _make_attention()
    features, hidden = _make_inputs()
    context, weights = attn(features, hidden)
    assert context.shape == (2, 2048)
    assert weights.shape == (2, 49)


def test_weights_sum_to_one():
    attn = _make_attention()
    features, hidden = _make_inputs()
    _, weights = attn(features, hidden)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_weights_non_negative():
    attn = _make_attention()
    features, hidden = _make_inputs()
    _, weights = attn(features, hidden)
    assert (weights >= 0).all()


def test_weights_finite():
    attn = _make_attention()
    features, hidden = _make_inputs()
    _, weights = attn(features, hidden)
    assert torch.isfinite(weights).all()


def test_context_values_finite():
    attn = _make_attention()
    features, hidden = _make_inputs()
    context, _ = attn(features, hidden)
    assert torch.isfinite(context).all()


def test_context_is_weighted_sum():
    attn = _make_attention()
    features, hidden = _make_inputs(b=2, n=49, enc=2048, dec=512)
    context, weights = attn(features, hidden)
    expected = (weights.unsqueeze(2) * features).sum(dim=1)
    assert torch.allclose(context, expected, atol=1e-6)


def test_batch_dim_unchanged():
    attn = _make_attention()
    features = torch.randn(5, 49, 2048)
    hidden = torch.randn(5, 512)
    context, weights = attn(features, hidden)
    assert context.shape[0] == 5
    assert weights.shape[0] == 5


def test_gradient_flows_to_encoder():
    attn = _make_attention()
    features, hidden = _make_inputs()
    features.requires_grad_(True)
    hidden.requires_grad_(True)
    context, _ = attn(features, hidden)
    loss = context.sum()
    loss.backward()
    assert attn.encoder_att.weight.grad is not None
    assert attn.decoder_att.weight.grad is not None
    assert features.grad is not None
    assert hidden.grad is not None