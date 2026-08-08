import os
import types

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import ImageCaptioningDataset, collate_fn
from src.data.vocabulary import Vocabulary


def make_cfg(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "Images").mkdir(parents=True, exist_ok=True)
    splits = data_root / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    images = {
        "img1": [
            "A dog runs in the grass .",
            "The dog is running .",
            "A dog playing outside .",
            "A dog in a field .",
            "The dog runs .",
        ],
        "img2": [
            "A cat sleeps on a mat .",
            "The cat is sleeping .",
            "A cat rests on the mat .",
            "The mat has a cat on it .",
            "Cat sleeping on mat .",
        ],
        "img3": [
            "A bird flies in the sky .",
            "A bird is flying .",
            "The bird soars .",
            "A bird in the air .",
            "Birds flying high .",
        ],
    }
    for iid, caps in images.items():
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        img.save(data_root / "Images" / f"{iid}.jpg")
    cap_path = data_root / "captions.txt"
    with open(cap_path, "a", encoding="utf-8") as f:
        f.write("image,caption\n")
        for iid, caps in images.items():
            for c in caps:
                f.write(f"{iid}.jpg,\"{c}\"\n")

    (splits / "train.txt").write_text("\n".join(f"{i}.jpg" for i in images) + "\n")

    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(
            caption_max_len=12,
            image_size=224,
            splits_dir=str(splits),
            images_dir=str(data_root / "Images"),
            captions_file=str(cap_path),
        )
    )
    return cfg, data_root


@pytest.fixture()
def vocab():
    return Vocabulary(
        [
            "a", "dog", "runs", "the", "grass", "is", "cat", "sleeps", "on", "mat",
            "in", "flies", "sky", "bird", "field", "outside",
        ],
        min_freq=1,
    )


def test_dataset_items(tmp_path, vocab):
    cfg, _ = make_cfg(tmp_path)
    ds = ImageCaptioningDataset("train", cfg, vocab)
    assert len(ds) == 15
    item = ds[0]
    assert item["image"].shape == torch.Size([3, 224, 224])
    assert item["caption_ids"][0] == Vocabulary.SOS
    assert item["caption_ids"][-1] == Vocabulary.EOS
    assert len(item["caption_ids"]) <= cfg.data.caption_max_len
    assert item["image_id"].endswith(".jpg")
    assert 0 <= item["caption_index"] < 5


def test_collate_pads_uneven(tmp_path, vocab):
    cfg, _ = make_cfg(tmp_path)
    ds = ImageCaptioningDataset("train", cfg, vocab)
    batch = [ds[0], ds[5], ds[10]]
    out = collate_fn(batch)
    assert out["images"].shape[0] == 3
    assert out["images"].shape[1:] == torch.Size([3, 224, 224])
    assert out["caption_ids"].dim() == 2
    assert out["caption_ids"].shape[0] == 3
    max_len = out["caption_lengths"].max().item()
    assert out["caption_ids"].shape[1] == max_len


def test_collate_pad_token_is_zero(tmp_path, vocab):
    cfg, _ = make_cfg(tmp_path)
    ds = ImageCaptioningDataset("train", cfg, vocab)
    batch = [ds[0], ds[2]]
    out = collate_fn(batch)
    long_lens = out["caption_lengths"]
    for i, length in enumerate(long_lens):
        tail = out["caption_ids"][i, length:]
        assert (tail == Vocabulary.PAD).all()


def test_dataset_skips_missing_image(tmp_path, vocab):
    cfg, data_root = make_cfg(tmp_path)
    os.remove(data_root / "Images" / "img2.jpg")
    ds = ImageCaptioningDataset("train", cfg, vocab)
    assert len(ds) == 10