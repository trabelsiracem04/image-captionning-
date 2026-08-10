from src.evaluation.metrics import bleu_score, bleu_summary


def test_perfect_match_bleu_one():
    tokens = ["a", "dog", "runs", "fast"]
    refs = [["a", "dog", "runs", "fast"]]
    scores = bleu_score(tokens, refs, max_n=4)
    assert scores[1] == 1.0
    assert scores[2] == 1.0
    assert scores[4] == 1.0


def test_disjoint_vocab_bleu_zero():
    tokens = ["cat", "sleeps"]
    refs = [["dog", "runs"]]
    scores = bleu_score(tokens, refs, max_n=4)
    assert scores[1] == 0.0


def test_brevity_penalty():
    tokens = ["a", "dog"]
    refs = [["a", "brown", "dog", "runs", "quickly"]]
    scores = bleu_score(tokens, refs, max_n=1)
    assert 0.0 < scores[1] < 1.0


def test_multi_reference_higher_than_single():
    cand = ["a", "dog", "on", "a", "couch"]
    weak_ref = [["a", "cat", "on", "a", "chair"]]
    strong_ref = [["a", "dog", "on", "a", "couch"], ["the", "dog", "lies", "on", "sofa"]]
    assert bleu_score(cand, strong_ref, max_n=1)[1] > bleu_score(cand, weak_ref, max_n=1)[1]


def test_empty_candidate_all_zero():
    scores = bleu_score([], [["a", "dog"]], max_n=4)
    assert all(scores[n] == 0.0 for n in range(1, 5))


def test_summary_averages():
    cands = [["a", "dog", "runs", "fast"], ["a", "dog", "runs", "fast"]]
    refs = [[["a", "dog", "runs", "fast"]], [["a", "dog", "runs", "fast"]]]
    summary = bleu_summary(cands, refs, max_n=4)
    assert summary[1] == 1.0
    assert summary[4] == 1.0


def test_partial_overlap_in_range():
    cand = ["a", "dog", "runs"]
    refs = [["a", "dog", "runs", "fast"]]
    scores = bleu_score(cand, refs, max_n=4)
    assert all(0.0 <= scores[n] <= 1.0 for n in range(1, 5))
    assert scores[1] > 0.0