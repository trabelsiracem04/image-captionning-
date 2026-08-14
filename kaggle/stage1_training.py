"""Paste this into a Kaggle Notebook cell to run stage-1 training.

Expects two Kaggle Datasets attached to the notebook:
  - "image-captioning-code"  (dist/kaggle_code.zip  -> mounts at /kaggle/input/image-captioning-code)
  - "image-captioning-data"  (dist/kaggle_data.zip  -> mounts at /kaggle/input/image-captioning-data)

Adjust the two SLUGS below if you named the datasets differently.
Edit EPOCHS as needed. Run as a single cell.
"""

import os
import shutil
import subprocess
from pathlib import Path

CODE_SLUG = "image-captioning-code"   # dataset name for the code zip
DATA_SLUG = "image-captioning-data"   # dataset name for the data zip
EPOCHS = 10

WORK = Path("/kaggle/working")
CODE_IN = Path("/kaggle/input") / CODE_SLUG
DATA_IN = Path("/kaggle/input") / DATA_SLUG
CODE_DIR = WORK / "imgcap"

# 1. path overrides -> train.py reads these via apply_env_overrides()
os.environ["IMG_CAP_DATA_ROOT"] = str(DATA_IN / "data")
os.environ["IMG_CAP_CHECKPOINTS_DIR"] = str(WORK / "checkpoints")
os.environ["IMG_CAP_OUTPUT_DIR"] = str(WORK / "outputs")
os.environ["IMG_CAP_NUM_WORKERS"] = "4"

print("input datasets:", CODE_IN.exists(), DATA_IN.exists())
assert CODE_IN.exists(), f"code dataset not found: {CODE_IN}"
assert DATA_IN.exists(), f"data dataset not found: {DATA_IN}"

# 2. copy code into writable space
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True, exist_ok=True)
for item in CODE_IN.iterdir():
    shutil.copytree(item, CODE_DIR / item.name) if item.is_dir() else shutil.copy2(item, CODE_DIR / item.name)
print("code ready at", CODE_DIR)

# 3. install lightweight deps (torch/torchvision already on Kaggle)
subprocess.run(["pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=CODE_DIR, check=True)

# 4. run training
subprocess.run(
    ["python", "train.py", "--tag", "stage1", "--epochs", str(EPOCHS)],
    cwd=CODE_DIR,
    check=True,
)

# 5. report checkpoints (auto-saved to notebook Output for download)
print("\n--- checkpoints (download these) ---")
for p in sorted((WORK / "checkpoints").glob("*.pt")):
    print(p)
print("log:", CODE_DIR / "experiments" / "runs" / "stage1")