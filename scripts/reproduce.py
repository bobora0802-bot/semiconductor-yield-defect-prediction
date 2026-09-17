"""One entry point: raw data, full study, executed notebooks and invariant tests."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for command in (["scripts/run_study.py"], ["scripts/build_study_notebooks.py"], ["-m", "pytest", "-q"], ["scripts/verify_artifacts.py"]):
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
