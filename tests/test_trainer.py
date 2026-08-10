import torch
from torch.utils.data import DataLoader

from src.data.vocabulary import Vocabulary
from src.models.caption_model import build_caption_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.checkpoints import load_checkpoint


def _make_vocab(words):
    return Vocabulary(words, min_freq=1)


def _make_config():
    cfg = load_config()
    cfg.model.encoder_pretrained = False
    cfg.training.fine_tune = False
    cfg.training.batch_size = 4
    cfg.training.epochs = 2
    cfg.training.patience = 1
    return cfg


def _iter_batches(batches):
    yield from batches


def _tiny_train_data(cfg, vocab):
    model = build_caption_model(cfg, vocab)
    images = torch.randn(4, 3, 224, 224)
    ids = torch.randint(1, len(vocab), (4, 6))
    ids[:, 0] = vocab.SOS
    ids[:, -1] = vocab.EOS
    batch = {
        "images": images,
        "caption_ids": ids,
        "image_ids": ["i1", "i2", "i3", "i4"],
    }
    return model, _iter_batches([batch])


def test_trainer_runs_cpu():
    cfg = _make_config()
    vocab = _make_vocab(["dog", "cat", "runs", "a", "the"])
    model, train_data = _tiny_train_data(cfg, vocab)
    trainer = Trainer(cfg, model, vocab, torch.device("cpu"), train_data)
    best = trainer.train(1)
    assert isinstance(best, float)


def test_encoder_stays_frozen_during_training():
    cfg = _make_config()
    vocab = _make_vocab(["dog", "cat", "runs", "a", "the"])
    model, train_data = _tiny_train_data(cfg, vocab)
    before = [p.clone() for p in model.encoder.parameters()]
    trainer = Trainer(cfg, model, vocab, torch.device("cpu"), train_data)
    trainer.train(1)
    for p, b in zip(model.encoder.parameters(), before):
        assert torch.equal(p.detach(), b)


def test_checkpoints_written_and_loadable(tmp_path):
    cfg = _make_config()
    vocab = _make_vocab(["dog", "cat", "runs", "a", "the"])
    model, train_data = _tiny_train_data(cfg, vocab)
    ckpt_dir = str(tmp_path)
    trainer = Trainer(
        cfg,
        model,
        vocab,
        torch.device("cpu"),
        train_data,
        val_loader=_iter_batches([]),
        checkpoints_dir=ckpt_dir,
    )
    trainer.train(1)
    from src.utils.checkpoints import checkpoint_paths

    best, last = checkpoint_paths(ckpt_dir)
    assert best.exists() and last.exists()

    model2, _ = _tiny_train_data(cfg, vocab)
    info = load_checkpoint(best, model2)
    assert "epoch" in info and "best_bleu1" in info["metrics"]


def test_resume_offsets_epoch(tmp_path):
    cfg = _make_config()
    vocab = _make_vocab(["dog", "cat", "runs", "a", "the"])
    model, train_data = _tiny_train_data(cfg, vocab)
    ckpt_dir = str(tmp_path / "ckpt")
    trainer = Trainer(cfg, model, vocab, torch.device("cpu"), train_data, checkpoints_dir=ckpt_dir)
    trainer.train(1)
    from src.utils.checkpoints import checkpoint_paths

    model2, _ = _tiny_train_data(cfg, vocab)
    trainer2 = Trainer(cfg, model2, vocab, torch.device("cpu"), train_data, checkpoints_dir=ckpt_dir)
    trainer2.resume(checkpoint_paths(ckpt_dir)[1])
    assert trainer2.start_epoch == 1


def test_early_stopping_returns():
    cfg = _make_config()
    cfg.training.patience = 1
    vocab = _make_vocab(["dog", "cat", "runs", "a", "the"])
    model, train_data = _tiny_train_data(cfg, vocab)
    refs = {"i1": [["dog", "runs"]]}

    trainer = Trainer(
        cfg,
        model,
        vocab,
        torch.device("cpu"),
        train_data,
        val_loader=_iter_batches([]),
        references_by_id=refs,
    )
    best = trainer.train(3)
    assert trainer.early_stopped is True