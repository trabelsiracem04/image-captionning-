"""Build Kaggle upload artifacts.

Usage (from repo root):
    python scripts/make_kaggle_bundle.py --code          # -> dist/kaggle_code.zip
    python scripts/make_kaggle_bundle.py --data          # -> dist/kaggle_data.zip
    python scripts/make_kaggle_bundle.py --upload-data   # -> dist/kaggle_data_upload/ (CLI upload folder)
    python scripts/make_kaggle_bundle.py                 # all three

The code zip is uploaded as a Kaggle Dataset via the web. The upload-data
folder is published with the Kaggle CLI:  kaggle datasets create -p dist/kaggle_data_upload
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Change to your Kaggle username (from your kaggle.json) and desired slug.
KAGGLE_USERNAME = "racemtrabelsi"
DATA_SLUG = "image-captioning-data"
IS_PRIVATE = True

CODE_ENTRIES = [
    ("train.py", "train.py"),
    ("src", "src"),
    ("configs", "configs"),
    ("requirements-kaggle.txt", "requirements-kaggle.txt"),
]

DATA_METADATA = {
    "id": f"{KAGGLE_USERNAME}/{DATA_SLUG}",
    "title": "Image Captioning Data (Flickr30k)",
    "subtitle": "Flickr30k images + captions + deterministic splits for stage-1 image captioning",
    "isPrivate": IS_PRIVATE,
    "licenses": [{"name": "other"}],
}


def _add_dir(zf: zipfile.ZipFile, src_dir: Path, arc_root: str) -> None:
    for path in sorted(src_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir)
        if "__pycache__" in rel.parts or rel.name.endswith(".pyc"):
            continue
        zf.write(path, f"{arc_root}/{rel.as_posix()}")


def build_code() -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "kaggle_code.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in CODE_ENTRIES:
            p = ROOT / src
            if p.is_dir():
                _add_dir(zf, p, arc)
            else:
                zf.write(p, arc)
    return out


def build_data() -> Path:
    data_root = ROOT / "data"
    if not data_root.exists():
        raise SystemExit("data/ not found - nothing to bundle.")
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "kaggle_data.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _add_dir(zf, data_root, "data")
    return out


def build_upload_dir() -> Path:
    data_root = ROOT / "data"
    if not data_root.exists():
        raise SystemExit("data/ not found - nothing to upload.")
    out = DIST / "kaggle_data_upload"
    if out.exists():
        shutil.rmtree(out)
    target = out / "data"
    shutil.copytree(data_root, target)
    with open(out / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(DATA_METADATA, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", action="store_true", help="build kaggle_code.zip only")
    parser.add_argument("--data", action="store_true", help="build kaggle_data.zip only")
    parser.add_argument("--upload-data", action="store_true", help="build dist/kaggle_data_upload/ for CLI upload")
    args = parser.parse_args()

    if not (args.code or args.data or args.upload_data):
        args.code = args.data = args.upload_data = True

    if args.code:
        out = build_code()
        print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB)")

    if args.data:
        out = build_data()
        print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB)")

    if args.upload_data:
        out = build_upload_dir()
        total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"wrote {out} ({total / 2**20:.2f} MiB unpacked)")
        print("publish with:  kaggle datasets create -p dist/kaggle_data_upload")


if __name__ == "__main__":
    main()