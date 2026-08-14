import os

import pytest

from src.utils.config import apply_env_overrides, load_config

ENV_KEYS = [
    "IMG_CAP_DATA_ROOT",
    "IMG_CAP_CHECKPOINTS_DIR",
    "IMG_CAP_OUTPUT_DIR",
    "IMG_CAP_NUM_WORKERS",
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_unset():
    cfg = apply_env_overrides(load_config())
    assert cfg.paths.data_root == "data"
    assert cfg.paths.checkpoints_dir == "checkpoints"
    assert cfg.paths.output_dir == "outputs"
    assert cfg.data.num_workers == 2


def test_env_overrides_paths(monkeypatch):
    monkeypatch.setenv("IMG_CAP_DATA_ROOT", "/kaggle/input/imgcap-data/data")
    monkeypatch.setenv("IMG_CAP_CHECKPOINTS_DIR", "/kaggle/working/checkpoints")
    monkeypatch.setenv("IMG_CAP_OUTPUT_DIR", "/kaggle/working/outputs")
    monkeypatch.setenv("IMG_CAP_NUM_WORKERS", "4")
    cfg = apply_env_overrides(load_config())
    assert cfg.paths.data_root == "/kaggle/input/imgcap-data/data"
    assert cfg.paths.checkpoints_dir == "/kaggle/working/checkpoints"
    assert cfg.paths.output_dir == "/kaggle/working/outputs"
    assert cfg.data.num_workers == 4


def test_env_num_workers_must_be_int(monkeypatch):
    monkeypatch.setenv("IMG_CAP_NUM_WORKERS", "many")
    with pytest.raises(ValueError):
        apply_env_overrides(load_config())


def test_partial_env_overrides(monkeypatch):
    monkeypatch.setenv("IMG_CAP_NUM_WORKERS", "6")
    cfg = apply_env_overrides(load_config())
    assert cfg.data.num_workers == 6
    assert cfg.paths.data_root == "data"
    assert cfg.paths.checkpoints_dir == "checkpoints"