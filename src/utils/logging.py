import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def make_logger(name="image_captionning", run_dir: str | Path | None = None):
    """Returns (logger, log_dir). Logs to console and to log_dir/train.log."""
    log_dir = Path(run_dir) if run_dir is not None else None
    if log_dir is None:
        log_dir = PROJECT_ROOT / "experiments" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, log_dir