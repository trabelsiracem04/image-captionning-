import torch


def get_device(override: str | None = None) -> torch.device:
    if override is not None and override != "auto":
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index or 0} ({torch.cuda.get_device_name(device)})"
    return "cpu"