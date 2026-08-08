from src.data.tokenizer import normalize, tokenize


def test_normalize_lowercase_and_whitespace():
    assert normalize("  HeLLo   World  ") == "hello world"


def test_normalize_curly_quotes():
    assert normalize("\u201CHe\u2019s here\u201D") == '"he\'s here"'


def test_tokenize_simple():
    assert tokenize("A dog runs .") == ["a", "dog", "runs"]


def test_tokenize_keeps_interior_apostrophe():
    assert tokenize("the man's hat") == ["the", "man's", "hat"]


def test_tokenize_strips_standalone_punctuation():
    assert tokenize("cars - bikes, .") == ["cars", "bikes"]


def test_tokenize_strips_edges():
    assert tokenize('"hello"') == ["hello"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("...") == []