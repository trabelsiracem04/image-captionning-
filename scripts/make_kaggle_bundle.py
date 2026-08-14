"""Build Kaggle upload zips.

Usage (from repo root):
    python scripts/make_kaggle_bundle.py --code     # -> dist/kaggle_code.zip
    python scripts/make_kaggle_bundle.py --data     # -> dist/kaggle_data.zip
    python scripts/make_kaggle_bundle.py            # both

The two zips are uploaded as two separate Kaggle Datasets.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

CODE_ENTRIES = [
    ("train.py", "train.py"),
    ("src", "src"),
    ("configs", "configs"),
    ("requirements-kaggle.txt", "requirements-kaggle.txt"),
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", action="store_true", help="build kaggle_code.zip only")
    parser.add_argument("--data", action="store_true", help="build kaggle_data.zip only")
    args = parser.parse_args()

    if not args.code and not args.data:
        args.code = args.data = True

    if args.code:
        out = build_code()
        print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB)")

    if args.data:
        out = build_data()
        print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()