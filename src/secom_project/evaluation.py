"""Evaluation helpers for temporal robustness, uncertainty, and screening policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ChronologicalSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    time_train: pd.Series
    time_test: pd.Series


def chronological_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    test_size: float = 0.20,
) -> ChronologicalSplit:
    """Use the latest observations as a deployment-style holdout."""

    if not len(X) == len(y) == len(timestamps):
        raise ValueError("X, y, and timestamps must have equal length.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    test_rows = max(1, int(np.ceil(len(X) * test_size)))
    split_at = len(X) - test_rows
    train_index = order[:split_at]
    test_index = order[split_at:]
    if y.iloc[train_index].nunique() < 2 or y.iloc[test_index].nunique() < 2:
        raise ValueError("Chronological train and test partitions must contain both classes.")
    return ChronologicalSplit(
        X_train=X.iloc[train_index].copy(),
        X_test=X.iloc[test_index].copy(),
        y_train=y.iloc[train_index].copy(),
        y_test=y.iloc[test_index].copy(),
        time_train=timestamps.iloc[train_index].copy(),
        time_test=timestamps.iloc[test_index].copy(),
    )


def expanding_window_probabilities(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    n_splits: int = 5,
    initial_train_fraction: float = 0.50,
) -> tuple[pd.Series, np.ndarray]:
    """Generate temporally ordered validation probabilities from expanding fits."""

    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1.")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    X_ordered = X.iloc[order]
    y_ordered = y.iloc[order]
    start = int(np.floor(len(X_ordered) * initial_train_fraction))
    if start < 1 or len(X_ordered) - start < n_splits:
        raise ValueError("Not enough rows for the requested expanding-window splits.")
    validation_blocks = np.array_split(np.arange(start, len(X_ordered)), n_splits)
    probabilities: list[np.ndarray] = []
    targets: list[pd.Series] = []
    for block in validation_blocks:
        train_end = int(block[0])
        fitted = clone(estimator).fit(X_ordered.iloc[:train_end], y_ordered.iloc[:train_end])
        probabilities.append(fitted.predict_proba(X_ordered.iloc[block])[:, 1])
        targets.append(y_ordered.iloc[block])
    return pd.concat(targets), np.concatenate(probabilities)


def binary_metrics(
    y_true: Iterable[int], probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "brier_score": float(brier_score_loss(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "review_rate": float(np.mean(prediction)),
    }


def stratified_bootstrap_intervals(
    y_true: Iterable[int],
    probability: np.ndarray,
    threshold: float,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return percentile intervals while preserving holdout class counts."""

    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(probability, dtype=float)
    negative = np.flatnonzero(truth == 0)
    positive = np.flatnonzero(truth == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Bootstrap intervals require both classes.")
    rng = np.random.default_rng(random_state)
    metric_names = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "balanced_accuracy",
        "precision",
        "recall",
    ]
    distributions = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        sample = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        values = binary_metrics(truth[sample], score[sample], threshold)
        for name in metric_names:
            distributions[name].append(float(values[name]))
    alpha = 1 - confidence
    point = binary_metrics(truth, score, threshold)
    rows = []
    for name in metric_names:
        rows.append(
            {
                "metric": name,
                "estimate": float(point[name]),
                "lower": float(np.quantile(distributions[name], alpha / 2)),
                "upper": float(np.quantile(distributions[name], 1 - alpha / 2)),
                "confidence": confidence,
                "resamples": n_resamples,
            }
        )
    return pd.DataFrame(rows)


def capacity_table(
    y_true: Iterable[int],
    probability: np.ndarray,
    review_fractions: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> pd.DataFrame:
    """Measure failure capture when review capacity is limited to top-risk rows."""

    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(probability, dtype=float)
    order = np.argsort(-score, kind="stable")
    total_failures = int(truth.sum())
    baseline = float(truth.mean())
    rows = []
    for fraction in review_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Review fractions must be in (0, 1].")
        reviewed = max(1, int(np.ceil(len(truth) * fraction)))
        selected = order[:reviewed]
        captured = int(truth[selected].sum())
        precision = captured / reviewed
        rows.append(
            {
                "target_review_fraction": float(fraction),
                "reviewed_runs": reviewed,
                "actual_review_rate": reviewed / len(truth),
                "failures_captured": captured,
                "total_failures": total_failures,
                "failure_recall": captured / total_failures,
                "review_precision": precision,
                "lift_vs_baseline": precision / baseline,
                "false_alarms": reviewed - captured,
                "score_cutoff": float(score[selected[-1]]),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitive_threshold(
    y_true: Iterable[int],
    probability: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float = 1.0,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold on non-test predictions for an explicit cost ratio."""

    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Costs must be positive.")
    grid = np.asarray(
        list(thresholds) if thresholds is not None else np.linspace(0.01, 0.99, 99)
    )
    rows = []
    for threshold in grid:
        values = binary_metrics(y_true, probability, float(threshold))
        expected_cost = (
            false_negative_cost * int(values["false_negative"])
            + false_positive_cost * int(values["false_positive"])
        )
        rows.append({**values, "expected_cost": float(expected_cost)})
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["expected_cost", "false_negative", "false_positive", "threshold"],
        ascending=[True, True, True, False],
    ).iloc[0]
    return float(best["threshold"]), table

