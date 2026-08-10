import torch

from src.data.vocabulary import Vocabulary
from src.models.decoder import Decoder, build_decoder
from src.utils.config import load_config


def _make_vocab():
    return Vocabulary(["dog", "cat", "boy", "girl", "ball", "runs", "plays", ".", ","], min_freq=1)


def _make_decoder(vocab=None, **kwargs):
    vocab = vocab or _make_vocab()
    defaults = dict(
        encoder_dim=2048,
        embed_dim=256,
        hidden_dim=512,
        attention_dim=512,
        vocab_size=len(vocab),
        dropout=0.5,
    )
    defaults.update(kwargs)
    return Decoder(**defaults), vocab


def _make_inputs(b=2, t=12, n=49):
    enc = torch.randn(b, n, 2048)
    ids = torch.randint(0, 7, (b, t))
    ids[:, 0] = 1
    ids[:, -1] = 2
    return enc, ids


def test_embedding_shape():
    dec, vocab = _make_decoder()
    assert dec.embedding.weight.shape == (len(vocab), 256)


def test_init_hidden_state_shape():
    dec, _ = _make_decoder()
    enc = torch.randn(2, 49, 2048)
    h, c = dec.init_hidden_state(enc)
    assert h.shape == (2, 512)
    assert c.shape == (2, 512)


def test_forward_shapes():
    dec, vocab = _make_decoder()
    enc, ids = _make_inputs()
    logits, weights = dec(enc, ids)
    assert logits.shape == (2, 12, len(vocab))
    assert weights.shape == (2, 12, 49)


def test_logits_finite():
    dec, vocab = _make_decoder()
    enc, ids = _make_inputs()
    logits, _ = dec(enc, ids)
    assert torch.isfinite(logits).all()


def test_attention_weights_sum_to_one():
    dec, vocab = _make_decoder()
    enc, ids = _make_inputs()
    _, weights = dec(enc, ids)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 12), atol=1e-6)


def test_single_step_shapes():
    dec, vocab = _make_decoder()
    enc = torch.randn(3, 49, 2048)
    emb = torch.randn(3, 256)
    h0, c0 = dec.init_hidden_state(enc)
    logits, h1, c1, weights = dec.step(enc, emb, h0, c0)
    assert logits.shape == (3, len(vocab))
    assert h1.shape == (3, 512)
    assert c1.shape == (3, 512)
    assert weights.shape == (3, 49)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_dropout_off_in_eval_equals_safely():
    dec, vocab = _make_decoder()
    dec.eval()
    enc, ids = _make_inputs()
    with torch.no_grad():
        logits_a, _ = dec(enc, ids)
        logits_b, _ = dec(enc, ids)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_batch_independent_gradients():
    dec, vocab = _make_decoder()
    enc, ids = _make_inputs()
    logits, _ = dec(enc, ids)
    loss = logits.sum()
    loss.backward()
    assert dec.embedding.weight.grad is not None
    assert dec.lstm.weight_ih.grad is not None
    assert dec.fc.weight.grad is not None


def test_build_decoder_from_config():
    vocab = _make_vocab()
    dec = build_decoder(load_config(), vocab)
    assert dec.vocab_size == len(vocab)
    assert dec.embed_dim == 256
    assert dec.hidden_dim == 512
    assert dec.attention_dim == 512


def test_vocab_size_matches_embedding():
    vocab = _make_vocab()
    dec, _ = _make_decoder(vocab=vocab)
    assert dec.fc.out_features == len(vocab)
    assert dec.embedding.num_embeddings == len(vocab)