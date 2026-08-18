import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ImageCaptioningDataset, collate_fn
from src.data.split import load_captions, resolve_data_paths
from src.data.tokenizer import tokenize
from src.data.vocabulary import Vocabulary
from src.models.caption_model import build_caption_model
from src.training.trainer import Trainer
from src.utils.config import PROJECT_ROOT, apply_env_overrides, load_config
from src.utils.device import describe_device, get_device
from src.utils.logging import make_logger
from src.utils.seeds import set_all_seeds


def load_vocab(cfg, captions, split_dir):
    train_ids = split_dir / "train.txt"
    ids = [
        line.strip()
        for line in train_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokens = []
    for iid in ids:
        for caption in captions.get(iid, []):
            tokens.extend(tokenize(caption))
    min_freq = getattr(cfg.data, "min_freq", 1)
    return Vocabulary(tokens, min_freq=min_freq)


def build_loaders(cfg, vocab):
    paths = resolve_data_paths(cfg)
    captions = load_captions(paths["captions_file"])

    train_set = ImageCaptioningDataset("train", cfg, vocab)
    val_set = ImageCaptioningDataset("val", cfg, vocab) if (paths["splits_dir"] / "val.txt").exists() else None

    workers = getattr(cfg.data, "num_workers", 2)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    references_by_id = {}
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
        )
        val_ids = [
            line.strip()
            for line in (paths["splits_dir"] / "val.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        references_by_id = {
            iid: [tokenize(c) for c in captions.get(iid, [])] for iid in val_ids
        }
    return train_loader, val_loader, references_by_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Image captioning training (Stage 1 or Stage 2)")
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--tag", default=None, help="run name for the logs/checkpoints folder")
    parser.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    parser.add_argument("--checkpoints-dir", default=None, help="override checkpoints dir")
    parser.add_argument("--fine-tune", action="store_true", help="force CNN fine-tuning (Stage 2)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_env_overrides(cfg)
    set_all_seeds(cfg.seed)

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.checkpoints_dir is not None:
        cfg.paths.checkpoints_dir = args.checkpoints_dir
    if args.fine_tune:
        cfg.training.fine_tune = True

    device = get_device(cfg.device)
    log_dir = (PROJECT_ROOT / "experiments" / "runs" / args.tag) if args.tag else None
    logger, run_dir = make_logger(run_dir=log_dir)

    paths = resolve_data_paths(cfg)
    captions = load_captions(paths["captions_file"])
    vocab = load_vocab(cfg, captions, paths["splits_dir"])
    logger.info("vocab size (min_freq=%s): %s", vocab.min_freq, len(vocab))

    train_loader, val_loader, references_by_id = build_loaders(cfg, vocab)
    logger.info("train items: %s | val items: %s", len(train_loader.dataset), len(val_loader.dataset) if val_loader else "n/a")

    model = build_caption_model(cfg, vocab).to(device)
    logger.info("device: %s", describe_device(device))
    logger.info(
        "model: encoder frozen=%s | decoder params=%.2fM",
        not cfg.training.fine_tune,
        sum(p.numel() for p in model.decoder.parameters()) / 1e6,
    )

    trainer = Trainer(
        cfg,
        model,
        vocab,
        device,
        train_loader,
        val_loader=val_loader,
        references_by_id=references_by_id,
        checkpoints_dir=cfg.paths.checkpoints_dir,
        logger=logger,
    )

    if args.resume:
        trainer.resume(args.resume)

    stage = 2 if cfg.training.fine_tune else 1
    logger.info("training stage-%s for %s epochs...", stage, cfg.training.epochs)
    best = trainer.train(cfg.training.epochs)
    logger.info("done. best val BLEU-1=%.4f%s", best, " (early stopped)" if trainer.early_stopped else "")
    return best


if __name__ == "__main__":
    main()