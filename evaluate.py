"""Compute BLEU-1..4 over a split (default: test) using a trained checkpoint.

Example:
    python evaluate.py --split test
    python evaluate.py --split val --beam --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data.split import load_captions, resolve_data_paths
from src.data.tokenizer import tokenize
from src.data.transforms import build_transforms
from src.data.vocabulary import Vocabulary
from src.evaluation.metrics import bleu_summary
from src.inference.beam_search import beam_search
from src.inference.greedy import greedy_search
from src.models.caption_model import build_caption_model
from src.utils.checkpoints import load_checkpoint
from src.utils.config import apply_env_overrides, load_config
from src.utils.device import get_device


def load_vocab(cfg):
    paths = resolve_data_paths(cfg)
    train_ids = [
        line.strip()
        for line in (paths["splits_dir"] / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    captions = load_captions(paths["captions_file"])
    tokens = []
    for iid in train_ids:
        for caption in captions.get(iid, []):
            tokens.extend(tokenize(caption))
    return Vocabulary(tokens, min_freq=getattr(cfg.data, "min_freq", 1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU on a split")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--greedy", action="store_true", help="use greedy decoding instead of beam search")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_env_overrides(cfg)
    device = get_device(args.device or cfg.device)
    print("device:", device)
    max_length = args.max_length or getattr(cfg.inference, "max_length", 30)

    vocab = load_vocab(cfg)
    model = build_caption_model(cfg, vocab).to(device)
    info = load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    print(f"loaded {args.checkpoint} (epoch={info.get('epoch')})")

    transform = build_transforms(cfg.data.image_size, augment=False)
    paths = resolve_data_paths(cfg)
    ids = [
        line.strip()
        for line in (paths["splits_dir"] / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    captions = load_captions(paths["captions_file"])
    images_dir = paths["images_dir"]
    special = {vocab.SOS, vocab.EOS, vocab.PAD, vocab.UNK}

    candidates, references = [], []
    skipped = 0
    with torch.no_grad():
        for iid in ids:
            p = images_dir / iid
            if not p.exists():
                skipped += 1
                continue
            img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            features = model.encode(img)
            if args.greedy:
                res = greedy_search(
                    model.decoder, features, sos_id=vocab.SOS, eos_id=vocab.EOS, max_length=max_length
                )
            else:
                res = beam_search(
                    model.decoder,
                    features,
                    sos_id=vocab.SOS,
                    eos_id=vocab.EOS,
                    beam_width=getattr(cfg.inference, "beam_size", 3),
                    max_length=max_length,
                )
            candidates.append([vocab.token_at(t) for t in res[0].tokens if t not in special])
            refs = [tokenize(c) for c in captions.get(iid, [])]
            references.append([r for r in refs if r])

    scores = bleu_summary(candidates, references, max_n=4)
    print(f"split={args.split} images={len(candidates)} skipped={skipped}")
    for n in range(1, 5):
        print(f"  BLEU-{n}: {scores[n]:.4f}")


if __name__ == "__main__":
    main()
