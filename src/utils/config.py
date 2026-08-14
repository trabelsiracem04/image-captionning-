import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_namespace(d) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _deep_namespace(value) if isinstance(value, dict) else value)
    return ns


def _deep_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: _deep_dict(getattr(obj, key)) for key in vars(obj)}
    if isinstance(obj, dict):
        return {key: _deep_dict(value) for key, value in obj.items()}
    return obj


DEFAULTS = {
    "seed": 42,
    "device": "auto",
    "paths": {
        "data_root": "data",
        "images_dir": "Images",
        "captions_file": "captions.txt",
        "splits_dir": "splits",
        "split_manifest": "split_manifest.json",
        "checkpoints_dir": "checkpoints",
        "output_dir": "outputs",
    },
"data": {
            "caption_max_len": 30,
            "image_size": 224,
            "min_freq": 1,
            "train_size": 29783,
            "val_size": 1000,
            "test_size": 1000,
            "split_seed": 42,
            "num_workers": 2,
        },
    "model": {
        "encoder": "resnet50",
        "encoder_pretrained": True,
        "embed_dim": 256,
        "hidden_dim": 512,
        "attention_dim": 512,
        "dropout": 0.5,
    },
    "training": {
        "batch_size": 32,
        "epochs": 10,
        "lr_decoder": 0.001,
        "lr_cnn": 0.00001,
        "grad_clip": 5.0,
        "fine_tune": False,
        "patience": 3,
    },
    "inference": {"max_length": 30, "beam_size": 3},
}

REQUIRED_KEYS = ["seed", "paths", "model"]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_path() -> Path:
    return PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | None = None) -> SimpleNamespace:
    if path is None:
        path = config_path()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    merged = _deep_merge(dict(DEFAULTS), raw)
    missing = [k for k in REQUIRED_KEYS if k not in merged]
    if missing:
        raise ValueError(f"config missing required key(s): {missing}")
    return _deep_namespace(merged)


def to_dict(cfg: SimpleNamespace) -> dict:
    return _deep_dict(cfg)


def apply_env_overrides(cfg: SimpleNamespace) -> SimpleNamespace:
    """Override config paths from environment (used on Kaggle). No-op when unset."""
    overrides = {
        "IMG_CAP_DATA_ROOT": ("paths", "data_root"),
        "IMG_CAP_CHECKPOINTS_DIR": ("paths", "checkpoints_dir"),
        "IMG_CAP_OUTPUT_DIR": ("paths", "output_dir"),
        "IMG_CAP_NUM_WORKERS": ("data", "num_workers"),
    }
    for var, (section, key) in overrides.items():
        value = os.environ.get(var)
        if value is None:
            continue
        section_obj = getattr(cfg, section, None)
        if section_obj is None:
            continue
        if var == "IMG_CAP_NUM_WORKERS":
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"{var} must be an integer, got {value!r}")
        setattr(section_obj, key, value)
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(json.dumps(to_dict(cfg), indent=2))