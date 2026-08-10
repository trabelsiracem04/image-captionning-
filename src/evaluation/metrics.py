"""Pure-Python BLEU scoring (no external NLP deps)."""

import math


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0))]


def _clipped_precision(tokens, references, n):
    candidate_grams = _ngrams(tokens, n)
    if not candidate_grams:
        return 0.0, 0.0
    max_ref_counts = {}
    for ref in references:
        counts = {}
        for gram in _ngrams(ref, n):
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            if count > max_ref_counts.get(gram, 0):
                max_ref_counts[gram] = count
    clipped = sum(min(candidate_grams.count(g), max_ref_counts.get(g, 0)) for g in set(candidate_grams))
    return clipped, len(candidate_grams)


def bleu_score(tokens, references, max_n=4):
    """BLEU-n scores for one candidate against up-to-N reference token lists.

    References: one or more tokenised human captions. Returns {n: score} for
    1..max_n following the standard modified-precision + brevity-penalty BLEU.
    """
    if not tokens:
        return {n: 0.0 for n in range(1, max_n + 1)}
    references = [r for r in references if r]
    if not references:
        return {n: 0.0 for n in range(1, max_n + 1)}

    candidate_len = len(tokens)
    ref_lens = [len(r) for r in references]
    best_ref_len = min(ref_lens, key=lambda rl: (abs(rl - candidate_len), rl))
    brevity = 1.0 if candidate_len > best_ref_len else math.exp(1.0 - best_ref_len / candidate_len)

    scores = {}
    for n in range(1, max_n + 1):
        clipped, total = _clipped_precision(tokens, references, n)
        precision = clipped / total if total else 0.0
        scores[n] = brevity * precision if precision > 0 else 0.0
    return scores


def bleu_summary(candidates, references, max_n=4):
    """Average BLEU scores over a list of (candidate, [references]) pairs."""
    sizes = {n: 0 for n in range(1, max_n + 1)}
    totals = {n: 0.0 for n in range(1, max_n + 1)}
    for cand, refs in zip(candidates, references):
        for n, score in bleu_score(cand, refs, max_n).items():
            totals[n] += score
            sizes[n] += 1
    return {n: (totals[n] / sizes[n] if sizes[n] else 0.0) for n in range(1, max_n + 1)}