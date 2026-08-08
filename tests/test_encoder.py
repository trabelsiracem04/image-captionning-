import torch

from src.models.encoder import ResNetEncoder, build_encoder
from src.utils.config import load_config


def _make_encoder(**kwargs):
    kwargs.setdefault("pretrained", False)
    return ResNetEncoder(**kwargs)


def test_no_network_needed():
    encoder = _make_encoder()
    cfg = load_config()
    assert cfg.model.encoder == "resnet50"
    assert isinstance(encoder, ResNetEncoder)


def test_forward_shape():
    x = torch.randn(2, 3, 224, 224)
    out = _make_encoder()(x)
    assert out.shape == (2, 2048, 7, 7)


def test_spatial_features_shape():
    x = torch.randn(2, 3, 224, 224)
    feats = _make_encoder().spatial_features(x)
    assert feats.shape == (2, 49, 2048)


def test_global_features_shape():
    x = torch.randn(2, 3, 224, 224)
    feats = _make_encoder().global_features(x)
    assert feats.shape == (2, 2048)


def test_spatial_mean_equals_global():
    x = torch.randn(4, 3, 224, 224)
    enc = _make_encoder()
    spatial = enc.spatial_features(x)
    global_feats = enc.global_features(x)
    assert torch.allclose(spatial.mean(dim=1), global_feats, atol=1e-6)


def test_values_finite():
    x = torch.randn(2, 3, 224, 224)
    feats = _make_encoder().spatial_features(x)
    assert torch.isfinite(feats).all()


def test_attribute_metadata():
    enc = _make_encoder()
    assert enc.feature_dim == 2048
    assert enc.grid_h == 7 and enc.grid_w == 7
    assert enc.num_spatial == 49


def test_freeze_disables_all_gradients():
    enc = _make_encoder()
    enc.freeze()
    assert not any(p.requires_grad for p in enc.parameters())
    assert enc.fine_tune is False


def test_unfreeze_layer4_only():
    enc = _make_encoder()
    enc.freeze()
    enc.unfreeze_layer4()
    assert enc.fine_tune is True
    trainable = [(n) for n, p in enc.backbone.named_parameters() if p.requires_grad]
    assert any(n.startswith("4.") for n in trainable)
    assert all(n.startswith("4.") for n in trainable)


def test_unsupported_encoder_raises():
    try:
        _make_encoder(encoder="resnet101")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported encoder")


def test_build_encoder_from_config():
    cfg = load_config()
    cfg.model.encoder_pretrained = False
    enc = build_encoder(cfg)
    assert enc.feature_dim == 2048


def test_without_pretrained_is_frozen_initially():
    enc = _make_encoder(pretrained=False)
    assert enc.fine_tune is False