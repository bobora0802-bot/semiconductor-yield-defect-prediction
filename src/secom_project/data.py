"""Load and validate the raw UCI SECOM files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_ROWS = 1_567
EXPECTED_FEATURES = 590


def load_secom(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return process measurements, binary failure labels, and timestamps.

    UCI encodes pass as ``-1`` and fail as ``1``. This project maps failure to
    the positive class (1), because missed failures are the costly outcome.
    """

    raw_dir = Path(raw_dir)
    data_path = raw_dir / "secom.data"
    labels_path = raw_dir / "secom_labels.data"
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing SECOM raw files. Run scripts/download_data.py first."
        )

    features = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["raw_label", "timestamp"],
    )

    if features.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            f"Unexpected feature shape {features.shape}; expected "
            f"({EXPECTED_ROWS}, {EXPECTED_FEATURES})."
        )
    if len(labels) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected label count: {len(labels)}")
    if set(labels["raw_label"].unique()) != {-1, 1}:
        raise ValueError("SECOM labels must be encoded as -1 and 1.")

    features.columns = [f"sensor_{index:03d}" for index in range(EXPECTED_FEATURES)]
    target = labels["raw_label"].eq(1).astype("int8").rename("failure")
    timestamps = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="raise"
    ).rename("timestamp")
    return features, target, timestamps

