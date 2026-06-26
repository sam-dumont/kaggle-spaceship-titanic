#!/usr/bin/env python3
"""Download Spaceship Titanic competition data via the Kaggle CLI.

Idempotent: if the CSVs already exist it does nothing, so it's safe to re-run.
Run from the repo root:

    python src/download.py
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "spaceship-titanic"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPECTED = ["train.csv", "test.csv", "sample_submission.csv"]


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)

    if all((DATA_DIR / name).exists() for name in EXPECTED):
        print(f"✓ Data already present in {DATA_DIR} — nothing to do.")
        return 0

    print(f"Downloading '{COMPETITION}' into {DATA_DIR} ...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
            check=True,
        )
    except FileNotFoundError:
        print("✗ `kaggle` CLI not found. Install with: pip install kaggle", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            f"✗ Download failed (exit {exc.returncode}). "
            "Did you accept the competition rules on the website?",
            file=sys.stderr,
        )
        return 1

    # `kaggle competitions download` delivers one zip; unzip then tidy up.
    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Unzipping {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(DATA_DIR)
        zip_path.unlink()

    missing = [name for name in EXPECTED if not (DATA_DIR / name).exists()]
    if missing:
        print(f"✗ Expected files still missing after download: {missing}", file=sys.stderr)
        return 1

    print("✓ Done. Files:")
    for name in EXPECTED:
        size = (DATA_DIR / name).stat().st_size
        print(f"   {name}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
