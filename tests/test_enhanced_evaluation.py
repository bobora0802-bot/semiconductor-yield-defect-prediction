import numpy as np
import pandas as pd

from secom_project.evaluation import (
    capacity_table,
    chronological_holdout,
    cost_sensitive_threshold,
    stratified_bootstrap_intervals,
)


def test_chronological_holdout_orders_time_and_preserves_classes():
    X = pd.DataFrame({"x": np.arange(20)})
    y = pd.Series([0, 1] * 10)
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
    split = chronological_holdout(X, y, timestamps, test_size=0.25)
    assert len(split.X_test) == 5
    assert split.time_train.max() < split.time_test.min()
    assert split.y_test.nunique() == 2


def test_capacity_table_is_monotone_in_captured_failures():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    table = capacity_table(y, score, review_fractions=[0.1, 0.3, 0.6])
    assert table["failures_captured"].is_monotonic_increasing
    assert table.iloc[0]["failures_captured"] == 1


def test_cost_threshold_penalises_missed_failures():
    y = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.45, 0.35, 0.40, 0.30, 0.20, 0.10])
    threshold, _ = cost_sensitive_threshold(
        y, score, false_negative_cost=10, thresholds=[0.3, 0.4, 0.5]
    )
    assert threshold == 0.3


def test_bootstrap_intervals_are_bounded_and_include_estimates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.7, 0.2, 0.8, 0.4, 0.3, 0.1, 0.05])
    table = stratified_bootstrap_intervals(
        y, score, threshold=0.5, n_resamples=100, random_state=1
    )
    assert (table["lower"] <= table["upper"]).all()
    assert set(table["metric"]) >= {"roc_auc", "pr_auc", "precision", "recall"}

