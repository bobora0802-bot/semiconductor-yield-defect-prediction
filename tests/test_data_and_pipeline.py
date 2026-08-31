from pathlib import Path

import numpy as np

from secom_project.data import load_secom
from secom_project.modeling import build_model_pipelines


ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_contract():
    X, y, timestamps = load_secom(ROOT / "data" / "raw")
    assert X.shape == (1_567, 590)
    assert y.value_counts().to_dict() == {0: 1_463, 1: 104}
    assert timestamps.notna().all()


def test_every_pipeline_fits_and_scores_probabilities():
    X, y, _ = load_secom(ROOT / "data" / "raw")
    sample = np.r_[np.flatnonzero(y.to_numpy() == 0)[:180], np.flatnonzero(y.to_numpy() == 1)[:40]]
    for model in build_model_pipelines(random_state=7).values():
        model.fit(X.iloc[sample], y.iloc[sample])
        probability = model.predict_proba(X.iloc[sample[:12]])[:, 1]
        assert probability.shape == (12,)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()

