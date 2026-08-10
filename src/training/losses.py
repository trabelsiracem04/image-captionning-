import torch
import torch.nn.functional as F


def captioning_loss(logits, labels, pad_id=0):
    """Masked teacher-forcing cross-entropy over a caption batch.

    logits [B, T, V] produced by consuming labels[..., 0:T]; each logit at
    position t predicts the token at position t+1, so labels are shifted by
    one. Positions holding <PAD> contribute nothing to the loss.

    Args:
        logits: [B, T, vocab_size].
        labels: [B, T] token ids (<SOS> ... <EOS>, PAD-padded).
        pad_id: token id ignored by the loss (default <PAD>=0).

    Returns:
        scalar loss (0.0 if no non-pad target remains after the shift).
    """
    if logits.shape[0] == 0 or labels.shape[0] == 0:
        return logits.sum() * 0.0

    shifted_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)

    if shifted_labels.numel() == 0:
        return logits.sum() * 0.0

    valid = shifted_labels != pad_id
    if not valid.any():
        return logits.sum() * 0.0

    return F.cross_entropy(
        shifted_logits[valid],
        shifted_labels[valid],
    )