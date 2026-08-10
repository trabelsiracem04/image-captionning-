import torch
import torch.nn.functional as F

from src.inference.greedy import CaptionResult


class _BeamNode:
    __slots__ = ("token", "log_prob", "prev", "h", "c", "attention")

    def __init__(self, token, log_prob, prev, h, c, attention):
        self.token = token
        self.log_prob = log_prob
        self.prev = prev
        self.h = h
        self.c = c
        self.attention = attention

    def path(self):
        node = self
        ids = []
        attentions = []
        while node.prev is not None:
            ids.append(node.token)
            attentions.append(node.attention)
            node = node.prev
        return ids[::-1], attentions[::-1]


@torch.no_grad()
def beam_search(
    decoder,
    encoder_features,
    sos_id=1,
    eos_id=2,
    beam_width=3,
    max_length=30,
):
    """Beam search caption generation for a batch of images.

    Returns one best caption per image, decoded independently.

    Args:
        decoder: module with decoder.embedding, decoder.init_hidden_state and
                 decoder.step matching src.models.decoder.Decoder.
        encoder_features: [B, N, encoder_dim].

    Returns:
        list[CaptionResult].
    """
    batch_size = encoder_features.shape[0]
    device = encoder_features.device
    results = []

    for b in range(batch_size):
        features = encoder_features[b : b + 1]
        h, c = decoder.init_hidden_state(features)
        start = _BeamNode(token=sos_id, log_prob=0.0, prev=None, h=h, c=c, attention=None)

        nodes = [start]
        terminated = []

        for _ in range(max_length):
            candidates = []
            for node in nodes:
                word_emb = decoder.embedding(
                    torch.tensor([node.token], dtype=torch.long, device=device)
                )
                logits, h_n, c_n, weights = decoder.step(features, word_emb, node.h, node.c)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_log_prob, top_tokens = log_probs.topk(beam_width)

                for j in range(beam_width):
                    token = int(top_tokens[j])
                    node_log_prob = node.log_prob + float(top_log_prob[j])
                    if token == eos_id:
                        ended = _BeamNode(token, node_log_prob, node, h_n, c_n, weights[0].detach())
                        terminated.append(ended)
                    else:
                        candidates.append(
                            _BeamNode(token, node_log_prob, node, h_n, c_n, weights[0].detach())
                        )

            if not candidates:
                break
            candidates.sort(key=lambda n: n.log_prob, reverse=True)
            nodes = candidates[:beam_width]

        best = None
        if terminated:
            best = max(terminated, key=lambda n: n.log_prob)
        else:
            best = max(nodes, key=lambda n: n.log_prob) if nodes else start

        tokens, attentions = best.path()
        results.append(
            CaptionResult(
                tokens=tokens,
                log_prob=best.log_prob if best is not start else 0.0,
                attention=attentions,
            )
        )

    return results