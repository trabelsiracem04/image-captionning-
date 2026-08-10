import torch

from src.inference.beam_search import beam_search


class StubDecoder(torch.nn.Module):
    """Stub whose context tracks step via hidden[...,0], so each beam path
    follows the same controlled sequence regardless of branching."""

    def __init__(self, seq_ids, vocab_size=20, num_spatial=49, enc_dim=2048):
        super().__init__()
        self.seq_ids = seq_ids
        self.vocab_size = vocab_size
        self.num_spatial = num_spatial
        self.embedding = torch.nn.Embedding(vocab_size, 8)

    def init_hidden_state(self, features):
        b = features.shape[0]
        h = torch.zeros(b, 32)
        h[:, 0] = -1.0
        return h, torch.zeros(b, 32)

    def step(self, encoder_features, word_embedding, h, c):
        step_idx = int(h[:, 0].item()) + 1
        idx = self.seq_ids[min(step_idx, len(self.seq_ids) - 1)]
        logits = torch.zeros(encoder_features.shape[0], self.vocab_size)
        logits[:, idx] = 5.0
        h2 = h.clone()
        h2[:, 0] = float(step_idx)
        return (
            logits,
            h2,
            c.clone(),
            torch.ones(encoder_features.shape[0], self.num_spatial) / self.num_spatial,
        )


def _features(b=1, n=49, enc=2048):
    return torch.randn(b, n, enc)


def test_beam_returns_controlled_sequence():
    seq = [4, 6, 2]
    stub = StubDecoder(seq)
    r = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=3, max_length=len(seq))[0]
    assert r.tokens == seq


def test_beam_batch():
    seq = [4, 2]
    stub = StubDecoder(seq)
    results = beam_search(stub, _features(b=2), sos_id=1, eos_id=2, beam_width=2, max_length=len(seq))
    assert len(results) == 2
    assert all(r.tokens == seq for r in results)


def test_beam_deterministic():
    seq = [4, 6, 2]
    stub = StubDecoder(seq)
    a = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=3, max_length=len(seq))
    b = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=3, max_length=len(seq))
    assert a[0].tokens == b[0].tokens


def test_beam_tokens_in_valid_range():
    stub = StubDecoder([4, 6, 2])
    r = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=5, max_length=3)[0]
    assert all(0 <= t < stub.vocab_size for t in r.tokens)
    assert r.tokens[-1] == 2


def test_beam_collects_attention():
    stub = StubDecoder([4, 2])
    r = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=2, max_length=2)[0]
    assert len(r.attention) == 2


def test_beam_log_prob_finite():
    stub = StubDecoder([4, 2])
    r = beam_search(stub, _features(), sos_id=1, eos_id=2, beam_width=2, max_length=2)[0]
    assert torch.isfinite(torch.tensor(r.log_prob))
    assert r.log_prob < 0.0