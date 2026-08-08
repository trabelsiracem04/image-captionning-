import pytest

from src.data.vocabulary import Vocabulary


def test_reserved_ids():
    v = Vocabulary()
    assert v.token2index["<PAD>"] == 0
    assert v.token2index["<SOS>"] == 1
    assert v.token2index["<EOS>"] == 2
    assert v.token2index["<UNK>"] == 3
    assert v.index2token[3] == "<UNK>"


def test_build_index_consistency():
    v = Vocabulary(["dog", "cat", "dog", "bird"], min_freq=1)
    assert len(v) == 7
    assert v.token2index["dog"] == 4
    assert v.index2token[v.token2index["dog"]] == "dog"


def test_deterministic_order():
    a = Vocabulary(["dog", "dog", "cat"], min_freq=1)
    b = Vocabulary(["dog", "dog", "cat"], min_freq=1)
    assert a.token2index == b.token2index


def test_min_freq_unk():
    v = Vocabulary(["dog", "dog", "cat"], min_freq=2)
    assert "dog" in v
    assert "cat" not in v
    assert v.index_of("cat") == Vocabulary.UNK


def test_encode_unk_fallback():
    v = Vocabulary(["dog"], min_freq=1)
    assert v.encode(["dog", "unknown_word"]) == [v.token2index["dog"], Vocabulary.UNK]


def test_wrap_sos_eos():
    v = Vocabulary(["dog"], min_freq=1)
    assert v.wrap(["dog"]) == [Vocabulary.SOS, v.token2index["dog"], Vocabulary.EOS]


def test_wrap_truncates_max_len():
    v = Vocabulary(["a", "b", "c", "d"], min_freq=1)
    ids = v.wrap(["a", "b", "c"], max_len=4)
    assert len(ids) == 4
    assert ids[0] == Vocabulary.SOS
    assert ids[-1] == Vocabulary.EOS


def test_decode_roundtrip():
    v = Vocabulary(["dog", "cat"], min_freq=1)
    assert v.decode([Vocabulary.UNK]) == ["<UNK>"]
    assert v.decode(v.wrap(["dog", "cat"]))[1:] == ["dog", "cat", "<EOS>"]


def test_serialization_roundtrip():
    v = Vocabulary(["dog", "cat", "dog"], min_freq=1)
    v2 = Vocabulary.from_dict(v.to_dict())
    for token, index in v.token2index.items():
        assert v2.token2index[token] == index
    assert v2.min_freq == v.min_freq


def test_word_count_excludes_special():
    v = Vocabulary(["dog", "cat"], min_freq=1)
    assert v.word_count() == 2