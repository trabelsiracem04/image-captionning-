import torch

from src.data.vocabulary import Vocabulary
from src.evaluation.metrics import bleu_score
from src.training.validate import validate_captions


class ControlledDecoder(torch.nn.Module):
    def __init__(self, seq_ids, vocab_size=20, num_spatial=49):
        super().__init__()
        self.seq_ids = seq_ids
        self.vocab_size = vocab_size
        self.num_spatial = num_spatial
        self.embedding = torch.nn.Embedding(vocab_size, 8)

    def init_hidden_state(self, features):
        b = features.shape[0]
        return torch.zeros(b, 32), torch.zeros(b, 32)

    def step(self, encoder_features, word_embedding, h, c):
        self._t = getattr(self, "_t", -1) + 1
        idx = self.seq_ids[min(self._t, len(self.seq_ids) - 1)]
        logits = torch.zeros(encoder_features.shape[0], self.vocab_size)
        logits[:, idx] = 5.0
        return (
            logits,
            h.clone(),
            c.clone(),
            torch.ones(encoder_features.shape[0], self.num_spatial) / self.num_spatial,
        )


class ControlledModel(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def encode(self, images):
        return torch.randn(images.shape[0], 49, 2048)


_VOCAB_WORDS = ["dog", "cat", "runs", "fast"]


def _batch(image_id):
    return {"images": torch.randn(1, 3, 224, 224), "image_ids": [image_id]}


def _iterate(batches):
    yield from batches


def test_validate_strips_eos_and_scores_perfect_match():
    vocab = Vocabulary(_VOCAB_WORDS, min_freq=1)
    dog = vocab.index_of("dog")
    runs = vocab.index_of("runs")

    model = ControlledModel(ControlledDecoder([dog, runs, vocab.EOS]))
    vocab_size = len(vocab)

    batches = [_batch("img1")]

    refs = {"img1": [["dog", "runs"]]}

    metrics, samples = validate_captions(
        model,
        _iterate(batches),
        vocab,
        torch.device("cpu"),
        refs,
        max_length=10,
    )

    assert samples[0]["caption"] == "dog runs"
    assert "EOS" not in samples[0]["caption"].split()
    assert metrics[1] == 1.0
    assert metrics[2] == 1.0


def test_validate_scores_zero_for_wrong_caption():
    vocab = Vocabulary(_VOCAB_WORDS, min_freq=1)
    cat = vocab.index_of("cat")
    fast = vocab.index_of("fast")

    model = ControlledModel(ControlledDecoder([cat, fast, vocab.EOS]))

    batches = [_batch("img2")]
    refs = {"img2": [["dog", "runs"]]}

    metrics, _ = validate_captions(
        model, _iterate(batches), vocab, torch.device("cpu"), refs, max_length=10
    )
    assert metrics[1] == 0.0


def test_validate_handles_missing_reference():
    vocab = Vocabulary(_VOCAB_WORDS, min_freq=1)
    dog = vocab.index_of("dog")
    model = ControlledModel(ControlledDecoder([dog, vocab.EOS]))
    batches = [_batch("img3")]
    metrics, _ = validate_captions(
        model, _iterate(batches), vocab, torch.device("cpu"), {}, max_length=10
    )
    assert metrics[1] == 0.0


def test_bleu_same_reference_direct_compare():
    vocab = Vocabulary(_VOCAB_WORDS, min_freq=1)
    dog, runs = vocab.index_of("dog"), vocab.index_of("runs")
    tokens = [vocab.token_at(t) for t in [dog, runs]] + ["<EOS>"]
    refs = [["dog", "runs"]]
    clean = [t for t in tokens if t not in ("<EOS>", "<SOS>", "<PAD>", "<UNK>")]
    dirty_score = bleu_score(tokens, refs, max_n=2)
    clean_score = bleu_score(clean, refs, max_n=2)
    assert dirty_score[1] < clean_score[1]