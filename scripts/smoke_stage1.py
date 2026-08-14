"""End-to-end smoke gate for stage-1 training on a tiny real-data slice.

Verifies on real images with the real pretrained encoder that the full
Trainer loop (data -> model -> training -> validation -> checkpoints -> resume)
works, without committing to a full multi-hour epoch.

Run from the repo root:
    python scripts/smoke_stage1.py [--device cuda] [--n-train 100] [--n-val 40] [--epochs 1]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import ImageCaptioningDataset, collate_fn
from src.data.split import load_captions, resolve_data_paths
from src.data.tokenizer import tokenize
from src.data.vocabulary import Vocabulary
from src.inference.greedy import greedy_search, tokens_to_text
from src.models.caption_model import build_caption_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seeds import set_all_seeds


def log(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    log("smoke: importing torch and config...")
    cfg = load_config()
    set_all_seeds(cfg.seed)
    cfg.training.batch_size = args.batch_size
    cfg.training.patience = 5
    cfg.model.encoder_pretrained = True
    cfg.data.num_workers = 0

    device = get_device(args.device)
    log(f"smoke: device = {device}")

    log("smoke: resolving data paths + captions...")
    paths = resolve_data_paths(cfg)
    captions = load_captions(paths["captions_file"])
    train_ids = [
        l.strip()
        for l in (paths["splits_dir"] / "train.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][: args.n_train]
    val_ids = [
        l.strip()
        for l in (paths["splits_dir"] / "val.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][: args.n_val]

    log("smoke: building vocabulary...")
    tokens = []
    for iid in train_ids:
        for c in captions.get(iid, []):
            tokens.extend(tokenize(c))
    vocab = Vocabulary(tokens, min_freq=1)
    log(f"smoke: vocab size = {len(vocab)}")

    log("smoke: building sliced datasets + loaders...")
    train_ds = ImageCaptioningDataset("train", cfg, vocab, image_ids=train_ids)
    val_ds = ImageCaptioningDataset("val", cfg, vocab, image_ids=val_ids)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    references_by_id = {iid: [tokenize(c) for c in captions.get(iid, [])] for iid in val_ids}
    log(f"smoke: train items={len(train_ds)} batches={len(train_loader)} | val items={len(val_ds)}")

    log("smoke: building caption model (pretrained ResNet50)...")
    model = build_caption_model(cfg, vocab).to(device)
    log(
        f"smoke: decoder params = {sum(p.numel() for p in model.decoder.parameters()) / 1e6:.2f}M"
    )

    ckpt_dir = ROOT / "experiments" / "smoke_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        cfg,
        model,
        vocab,
        device,
        train_loader,
        val_loader=val_loader,
        references_by_id=references_by_id,
        checkpoints_dir=str(ckpt_dir),
    )

    log(f"smoke: training {args.epochs} epoch(s)...")
    t0 = time.time()
    best = trainer.train(args.epochs)
    log(f"smoke: training finished in {time.time() - t0:.1f}s, best val BLEU-1 = {best:.4f}")

    assert (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists(), "checkpoints missing"
    log("smoke: checkpoints best.pt + last.pt written OK")

    model2 = build_caption_model(cfg, vocab).to(device)
    trainer2 = Trainer(cfg, model2, vocab, device, train_loader, checkpoints_dir=str(ckpt_dir))
    trainer2.resume(ckpt_dir / "last.pt")
    log(f"smoke: resume OK (start_epoch = {trainer2.start_epoch})")
    assert trainer2.start_epoch == args.epochs, "resume start_epoch mismatch"

    log("smoke: greedy sample on a real val image...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        feats = model.encode(batch["images"].to(device))
        result = greedy_search(
            model.decoder,
            feats,
            sos_id=vocab.SOS,
            eos_id=vocab.EOS,
            max_length=cfg.inference.max_length,
        )[0]
    log(f"smoke: sample caption = {tokens_to_text(result.tokens, vocab)}")
    log("smoke: PASSED")


if __name__ == "__main__":
    main()