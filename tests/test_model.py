import torch

from src.data.vocabulary import Vocabulary
from src.models.caption_model import CaptionModel, build_caption_model
from src.utils.config import load_config


def _make_vocab():
    return Vocabulary(["dog", "cat", "boy", "girl", "a", "the"], min_freq=1)


def _offline_config():
    cfg = load_config()
    cfg.model.encoder_pretrained = False
    cfg.training.fine_tune = False
    return cfg


def test_build_offline():
    cfg = _offline_config()
    model = build_caption_model(cfg, _make_vocab())
    assert isinstance(model, CaptionModel)


def test_forward_shapes():
    cfg = _offline_config()
    model = build_caption_model(cfg, _make_vocab())
    images = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 5, (2, 12))
    ids[:, 0] = 1
    ids[:, -1] = 2
    logits, weights, features = model(images, ids)
    assert logits.shape == (2, 12, len(_make_vocab()))
    assert weights.shape == (2, 12, 49)
    assert features.shape == (2, 49, 2048)


def test_frozen_default_encoder():
    cfg = _offline_config()
    model = build_caption_model(cfg, _make_vocab())
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.decoder.parameters())


def test_fine_tune_unfreezes_layer4_only():
    cfg = _offline_config()
    cfg.training.fine_tune = True
    model = build_caption_model(cfg, _make_vocab())
    trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable
    assert all("layer4." in n for n in trainable)


def test_forward_finite_and_grads():
    cfg = _offline_config()
    model = build_caption_model(cfg, _make_vocab())
    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    ids = torch.randint(0, 5, (2, 12))
    ids[:, 0] = 1
    ids[:, -1] = 2
    logits, _, _ = model(images, ids)
    assert torch.isfinite(logits).all()
    loss = logits.mean()
    loss.backward()
    assert model.decoder.fc.weight.grad is not None
    assert images.grad is not None


def test_encode_and_step_wiring():
    cfg = _offline_config()
    model = build_caption_model(cfg, _make_vocab())
    model.eval()
    images = torch.randn(1, 3, 224, 224)
    features = model.encode(images)
    assert features.shape == (1, 49, 2048)
    h, c = model.init_hidden_state(features)
    emb = model.decoder.embedding(torch.tensor([[1]]))
    logits, h2, c2, weights = model.step(features, emb[:, 0], h, c)
    assert logits.shape == (1, len(_make_vocab()))
    assert h2.shape == (1, 512) and c2.shape == (1, 512)