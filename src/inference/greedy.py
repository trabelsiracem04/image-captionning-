from dataclasses import dataclass, field

import torch


@dataclass
class CaptionResult:
    tokens: list
    log_prob: float
    attention: list = field(default_factory=list)


def tokens_to_text(tokens, vocab) -> str:
    words = []
    for token in tokens:
        if token in (vocab.SOS, vocab.EOS, vocab.PAD, vocab.UNK):
            continue
        words.append(vocab.token_at(token))
    return " ".join(words)


@torch.no_grad()
def greedy_search(
    decoder,
    encoder_features,
    sos_id=1,
    eos_id=2,
    max_length=30,
):
    """Greedy (argmax) caption generation for a batch of images.

    Args:
        decoder: module with decoder.embedding, decoder.init_hidden_state and
                 decoder.step matching src.models.decoder.Decoder.
        encoder_features: [B, N, encoder_dim].

    Returns:
        list[CaptionResult] one per image (tokens include trailing <EOS>).
    """
    batch_size = encoder_features.shape[0]
    device = encoder_features.device

    h, c = decoder.init_hidden_state(encoder_features)

    current = torch.full((batch_size,), sos_id, dtype=torch.long, device=device)
    log_probs = [0.0] * batch_size
    tokens = [[] for _ in range(batch_size)]
    attention = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _ in range(max_length):
        word_emb = decoder.embedding(current)
        logits, h, c, weights = decoder.step(encoder_features, word_emb, h, c)

        probs = torch.softmax(logits, dim=-1)
        token_t = torch.argmax(probs, dim=-1)

        for b in range(batch_size):
            if finished[b]:
                continue
            tokens[b].append(int(token_t[b]))
            attention[b].append(weights[b].detach())
            step_log_prob = float(torch.log(probs[b, token_t[b]]).item())
            log_probs[b] += step_log_prob
            finished[b] = int(token_t[b]) == eos_id

        current = token_t

    results = []
    for b in range(batch_size):
        if not finished[b]:
            tokens[b].append(eos_id)
        results.append(
            CaptionResult(tokens=tokens[b], log_prob=log_probs[b], attention=attention[b])
        )
    return results