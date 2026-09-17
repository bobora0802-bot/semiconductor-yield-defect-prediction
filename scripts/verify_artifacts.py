"""Verify current code/input/artifact hashes against the generated study manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/study"


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".py", ".md", ".json", ".csv", ".txt", ".yml", ".ipynb"}:
        data = data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    count = 0
    for section, base in (("source_sha256", ROOT), ("data_sha256", ROOT/"data/raw"), ("artifact_sha256", OUT)):
        for name, expected in manifest[section].items():
            path = base / name
            if not path.exists() or digest(path) != expected:
                raise SystemExit(f"Manifest mismatch: {section}/{name}. Rebuild after reviewing the change.")
            count += 1
    print(f"Verified {count} source, input and artifact hashes.")


if __name__ == "__main__":
    main()
