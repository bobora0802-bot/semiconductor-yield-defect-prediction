"""Download the official UCI SECOM archive and verify its contents."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path


SOURCE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
EXPECTED_FILES = {"secom.data", "secom_labels.data", "secom.names"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "secom.zip"
    if not archive.exists():
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        missing = EXPECTED_FILES - names
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {sorted(missing)}")
        zipped.extractall(RAW_DIR)
    print(f"SECOM files ready in {RAW_DIR}")


if __name__ == "__main__":
    main()

