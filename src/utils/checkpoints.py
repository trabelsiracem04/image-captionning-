from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, epoch=0, stage=1, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    payload = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {
        "epoch": payload.get("epoch", 0),
        "stage": payload.get("stage", 1),
        "metrics": payload.get("metrics", {}),
    }


def checkpoint_paths(checkpoints_dir):
    """Return (best, last) checkpoint file paths under checkpoints_dir."""
    base = Path(checkpoints_dir)
    return base / "best.pt", base / "last.pt"