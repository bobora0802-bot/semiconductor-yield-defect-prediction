import numpy as np
import pandas as pd

from secom_project.spc import MultivariateSPC, UnivariateSPC, alert_metrics, define_phases


def _data(seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(100, 12)), columns=[f"x{i}" for i in range(12)])
    y = pd.Series(([0] * 40 + [1] * 10) * 2)
    return X, y


def test_reference_is_strictly_before_final_holdout():
    _, y = _data()
    phases = define_phases(y, train_size=80, phase_i_fraction=0.60)
    assert phases.reference_positions.max() < phases.phase_i_size < phases.train_size
    assert set(phases.phase_labels[80:]) == {"final_holdout"}


def test_control_limits_use_reference_only_and_ignore_holdout_labels():
    X, y = _data()
    phases_a = define_phases(y, train_size=80)
    flipped = y.copy()
    flipped.iloc[80:] = 1 - flipped.iloc[80:]
    phases_b = define_phases(flipped, train_size=80)
    reference_a = X.iloc[phases_a.reference_positions]
    reference_b = X.iloc[phases_b.reference_positions]
    first = MultivariateSPC().fit(reference_a)
    second = MultivariateSPC().fit(reference_b)
    assert first.t2_limit_ == second.t2_limit_
    assert first.q_limit_ == second.q_limit_
    assert np.array_equal(phases_a.reference_positions, phases_b.reference_positions)


def test_ewma_cusum_lengths_and_mspc_scores_are_valid():
    X, _ = _data()
    uni = UnivariateSPC().fit(X.iloc[:40])
    uni_scores = uni.transform(X)
    multi = MultivariateSPC().fit(X.iloc[:40])
    scores = multi.transform(X)
    assert len(uni_scores["ewma_alert"]) == len(X)
    assert len(uni_scores["cusum_alert"]) == len(X)
    assert np.isfinite(scores[["t2", "q_spe"]].to_numpy()).all()
    assert (scores[["t2", "q_spe"]].to_numpy() >= 0).all()


def test_pass_runs_between_false_alerts_counts_unflagged_passes():
    truth = np.zeros(5, dtype=int)
    assert alert_metrics(truth, np.ones(5, dtype=bool))["pass_runs_between_false_alerts"] == 0.0
    assert alert_metrics(truth, np.zeros(5, dtype=bool))["pass_runs_between_false_alerts"] == 5.0
    assert alert_metrics(truth, np.array([1, 0, 0, 1, 0], dtype=bool))["pass_runs_between_false_alerts"] == 2.0
