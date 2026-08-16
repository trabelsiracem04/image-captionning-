"""Paste this into a Kaggle Notebook cell to run stage-1 training.

Datasets are auto-located by marker files, so their Kaggle slugs/owner
don't matter. Expects two Kaggle Datasets attached to the notebook:
  - code  (contains train.py + src/)
  - data  (contains captions.txt + Images/)

Optional manual overrides (set as env vars before running):
  IMG_CAP_CODE_SLUG   -> pin the code dataset slug
  IMG_CAP_DATA_SLUG   -> pin the data dataset slug
  IMG_CAP_DATA_ROOT   -> pin the data root directly
"""

import os
import shutil
import subprocess
from pathlib import Path

EPOCHS = 10

WORK = Path("/kaggle/working")
CODE_DIR = WORK / "imgcap"


def _find_dataset(start: Path, marker_file: str, marker_dir: str):
    """Recursively find the dataset root containing marker_file + marker_dir/."""
    for f in start.rglob(marker_file):
        base = f.parent
        if (base / marker_dir).is_dir():
            return base
    return None


# 1. locate datasets (Kaggle nests them under /kaggle/input/[datasets/<owner>/]<slug>)
CODE_SLUG = os.environ.get("IMG_CAP_CODE_SLUG")
DATA_SLUG = os.environ.get("IMG_CAP_DATA_SLUG")

CODE_IN = None
if CODE_SLUG:
    cand = Path("/kaggle/input") / CODE_SLUG
    if cand.exists():
        CODE_IN = cand
if CODE_IN is None:
    CODE_IN = _find_dataset(Path("/kaggle/input"), "train.py", "src")

DATA_ROOT = os.environ.get("IMG_CAP_DATA_ROOT")
if DATA_ROOT:
    DATA_ROOT = Path(DATA_ROOT)
else:
    DATA_ROOT = _find_dataset(Path("/kaggle/input"), "captions.txt", "Images")

assert CODE_IN is not None and CODE_IN.exists(), (
    "code dataset (train.py + src/) not found under /kaggle/input"
)
assert DATA_ROOT is not None and DATA_ROOT.exists(), (
    "data dataset (captions.txt + Images/) not found under /kaggle/input"
)

print("code dataset at:", CODE_IN)
print("data root at   :", DATA_ROOT)

# 2. path overrides -> train.py reads these via apply_env_overrides()
os.environ["IMG_CAP_DATA_ROOT"] = str(DATA_ROOT)
os.environ["IMG_CAP_CHECKPOINTS_DIR"] = str(WORK / "checkpoints")
os.environ["IMG_CAP_OUTPUT_DIR"] = str(WORK / "outputs")
os.environ["IMG_CAP_NUM_WORKERS"] = "4"

# 3. copy code into writable space
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True, exist_ok=True)
for item in CODE_IN.iterdir():
    if item.is_dir():
        shutil.copytree(item, CODE_DIR / item.name)
    else:
        shutil.copy2(item, CODE_DIR / item.name)
print("code ready at", CODE_DIR)

# 4. patch Python 3.10+ incompat: collections.Iterable was removed
vp = CODE_DIR / "src/data/vocabulary.py"
s = vp.read_text(encoding="utf-8")
if "collections.Iterable" in s:
    vp.write_text(s.replace("collections.Iterable", "collections.abc.Iterable"), encoding="utf-8")
    print("patched collections.Iterable -> collections.abc.Iterable")

# 5. install lightweight deps (torch/torchvision already on Kaggle)
subprocess.run(["pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=CODE_DIR, check=True)

# 6. run training
subprocess.run(
    ["python", "train.py", "--tag", "stage1", "--epochs", str(EPOCHS)],
    cwd=CODE_DIR,
    check=True,
)

# 7. report checkpoints (auto-saved to notebook Output for download)
print("\n--- checkpoints (download these) ---")
for p in sorted((WORK / "checkpoints").glob("*.pt")):
    print(p)
print("log:", CODE_DIR / "experiments" / "runs" / "stage1")
