"""Generate captions for images with a trained Stage-1 checkpoint.

Examples:
    python predict.py --image path/to/img.jpg
    python predict.py --images-dir data/Images --beam
    python predict.py --split test --checkpoint checkpoints/best.pt
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
from src.inference.greedy import greedy_search, tokens_to_text
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


def build_model(cfg, vocab, checkpoint, device):
    model = build_caption_model(cfg, vocab).to(device)
    info = load_checkpoint(checkpoint, model, device=device)
    model.eval()
    return model, info


def caption_image(model, image_tensor, vocab, device, beam, max_length):
    with torch.no_grad():
        features = model.encode(image_tensor.unsqueeze(0).to(device))
        if beam:
            results = beam_search(
                model.decoder, features, sos_id=vocab.SOS, eos_id=vocab.EOS, max_length=max_length
            )
        else:
            results = greedy_search(
                model.decoder, features, sos_id=vocab.SOS, eos_id=vocab.EOS, max_length=max_length
            )
    return tokens_to_text(results[0].tokens, vocab)


def load_tensor(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def main():
    parser = argparse.ArgumentParser(description="Generate captions with a trained model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", default=None, help="path to a single image")
    parser.add_argument("--images-dir", default=None, help="directory of images to caption")
    parser.add_argument("--split", default=None, help="caption all image_ids in a split file")
    parser.add_argument("--beam", action="store_true", help="use beam search instead of greedy")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_env_overrides(cfg)
    device = get_device(args.device or cfg.device)
    max_length = args.max_length or getattr(cfg.inference, "max_length", 30)

    vocab = load_vocab(cfg)
    model, info = build_model(cfg, vocab, args.checkpoint, device)
    print(
        f"loaded {args.checkpoint} "
        f"(epoch={info.get('epoch')}, bleu1={info.get('metrics', {}).get('best_bleu1')})"
    )

    transform = build_transforms(cfg.data.image_size, augment=False)

    if args.image:
        t = load_tensor(args.image, transform)
        print(f"{args.image}: {caption_image(model, t, vocab, device, args.beam, max_length)}")
        return

    if args.images_dir:
        paths = sorted(Path(args.images_dir).rglob("*.jpg")) + sorted(
            Path(args.images_dir).rglob("*.jpeg")
        )
        for p in paths:
            t = load_tensor(p, transform)
            print(f"{p.name}: {caption_image(model, t, vocab, device, args.beam, max_length)}")
        return

    if args.split:
        paths = resolve_data_paths(cfg)
        ids = [
            line.strip()
            for line in (paths["splits_dir"] / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for iid in ids:
            p = paths["images_dir"] / iid
            if not p.exists():
                continue
            t = load_tensor(p, transform)
            print(f"{iid}: {caption_image(model, t, vocab, device, args.beam, max_length)}")
        return

    parser.error("provide --image, --images-dir, or --split")


if __name__ == "__main__":
    main()
