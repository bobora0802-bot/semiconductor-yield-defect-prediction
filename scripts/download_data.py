"""Download the official UCI SECOM archive and verify its contents."""

from __future__ import annotations

import urllib.request
import zipfile
import hashlib
from pathlib import Path


SOURCE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
EXPECTED_FILES = {"secom.data", "secom_labels.data", "secom.names"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_SHA256 = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
    "secom.names": "6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "secom.zip"
    if not archive.exists():
        print(f"Downloading {SOURCE_URL}")
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        missing = EXPECTED_FILES - names
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {sorted(missing)}")
        # Extract only known members and verify bytes before accepting them.
        contents = {name: zipped.read(name) for name in EXPECTED_FILES}
        for name, content in contents.items():
            if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256[name]:
                raise RuntimeError(f"Data checksum changed for {name}; inspect source before accepting a new dataset.")
        for name, content in contents.items():
            (RAW_DIR / name).write_bytes(content)
    print(f"SECOM files ready in {RAW_DIR}")


if __name__ == "__main__":
    main()
