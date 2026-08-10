import torch

from src.evaluation.metrics import bleu_summary
from src.inference.greedy import greedy_search


@torch.no_grad()
def validate_captions(
    model,
    dataloader,
    vocab,
    device,
    references_by_id,
    max_length=30,
    max_samples=None,
):
    """Greedy-decode a set and compute BLEU against the stored references.

    Args:
        model: CaptionModel (encoder + decoder), moved to `device` in eval mode.
        dataloader: iterable of {"images","image_ids"}-style batches.
        vocab: Vocabulary used to decode ids.
        device: torch device.
        references_by_id: {image_id: [[tokens], ...]} human captions.
        max_length: longest generated caption.
        max_samples: optional cap on images scored.

    Returns:
        (metrics dict bleu_1..bleu_4, samples list of caption dicts).
    """
    model.eval()
    model = model.to(device)

    special = {vocab.SOS, vocab.EOS, vocab.PAD, vocab.UNK}

    def clean(tokens):
        return [vocab.token_at(t) for t in tokens if t not in special]

    hyps = []
    refs = []
    samples = []

    for batch in dataloader:
        if max_samples is not None and len(hyps) >= max_samples:
            break

        images = batch["images"].to(device)
        image_ids = batch["image_ids"]

        features = model.encode(images)
        results = greedy_search(
            model.decoder,
            features,
            sos_id=vocab.SOS,
            eos_id=vocab.EOS,
            max_length=max_length,
        )

        for iid, r in zip(image_ids, results):
            hyp = clean(r.tokens)
            hyps.append(hyp)
            refs.append(references_by_id.get(iid, []))
            samples.append({"image_id": iid, "caption": " ".join(hyp)})

    metrics = bleu_summary(hyps, refs)
    return metrics, samples