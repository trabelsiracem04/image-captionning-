import math

import torch

from src.data.vocabulary import Vocabulary
from src.inference.greedy import greedy_search, tokens_to_text


class StubDecoder(torch.nn.Module):
    """Decoder stub whose argmax per step is fully controllable."""

    def __init__(self, seq_ids, vocab_size=20, num_spatial=49, enc_dim=2048):
        super().__init__()
        self.seq_ids = seq_ids
        self.vocab_size = vocab_size
        self.num_spatial = num_spatial
        self.enc_dim = enc_dim
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self._t = -1

    def init_hidden_state(self, features):
        b = features.shape[0]
        return torch.zeros(b, 32), torch.zeros(b, 32)

    def step(self, encoder_features, word_embedding, h, c):
        self._t += 1
        idx = self.seq_ids[self._t]
        logits = torch.zeros(encoder_features.shape[0], self.vocab_size)
        logits[:, idx] = 5.0
        return (
            logits,
            h.clone(),
            c.clone(),
            torch.ones(encoder_features.shape[0], self.num_spatial) / self.num_spatial,
        )


def _features(b=1, n=49, enc=2048):
    return torch.randn(b, n, enc)


def test_greedy_follows_controlled_sequence():
    seq = [5, 6, 7, 2]
    stub = StubDecoder(seq)
    r = greedy_search(stub, _features(), sos_id=1, eos_id=2, max_length=len(seq))[0]
    assert r.tokens == seq


def test_greedy_batch_shape():
    seq = [3, 4, 2]
    stub = StubDecoder(seq)
    results = greedy_search(stub, _features(b=2), sos_id=1, eos_id=2, max_length=len(seq))
    assert len(results) == 2
    assert all(r.tokens == seq for r in results)


def test_greedy_cumulative_log_prob():
    seq = [5, 2]
    stub = StubDecoder(seq)
    V = stub.vocab_size - 1
    expected = 2 * math.log(math.exp(5.0) / (V + math.exp(5.0)))
    r = greedy_search(stub, _features(), sos_id=1, eos_id=2, max_length=len(seq))[0]
    assert math.isclose(r.log_prob, expected, rel_tol=1e-5)


def test_greedy_forces_eos_without_one():
    stub = StubDecoder([5, 6])
    r = greedy_search(stub, _features(), sos_id=1, eos_id=2, max_length=2)[0]
    assert r.tokens == [5, 6, 2]


def test_greedy_tokens_to_text():
    vocab = Vocabulary(["dog", "cat", "a", "the"], min_freq=1)
    four = vocab.index_of("dog")
    five = vocab.index_of("cat")
    text = tokens_to_text([1, four, five, 2, 0, 3], vocab)
    assert text == "dog cat"


def test_greedy_attention_collected():
    stub = StubDecoder([5, 2])
    r = greedy_search(stub, _features(), sos_id=1, eos_id=2, max_length=2)[0]
    assert len(r.attention) == 2