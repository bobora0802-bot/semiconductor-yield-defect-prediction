"""Small, auditable building blocks for retrospective time-based experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .modeling import build_model_pipelines


def time_boundary(timestamps: pd.Series, fraction: float) -> int:
    """Move an approximate row boundary left so equal timestamps stay together."""
    if not timestamps.is_monotonic_increasing or timestamps.isna().any():
        raise ValueError("Timestamps must be sorted and non-null.")
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must be in (0, 1].")
    boundary = min(len(timestamps), int(np.floor(len(timestamps) * fraction)))
    while 0 < boundary < len(timestamps) and timestamps.iloc[boundary - 1] == timestamps.iloc[boundary]:
        boundary -= 1
    if boundary == 0:
        raise ValueError("No nonempty training period at this boundary.")
    return boundary


def windows(timestamps, starts, ends):
    if len(starts) != len(ends):
        raise ValueError("Window start/end lengths differ.")
    result = []
    for a, b in zip(starts, ends, strict=True):
        start, end = time_boundary(timestamps, a), time_boundary(timestamps, b)
        if start >= end:
            raise ValueError("Empty evaluation window.")
        result.append((start, end))
    return result


def study_models(seed=42):
    original = build_model_pipelines(seed)
    no_selection = clone(original["random_forest"])
    no_selection.set_params(select="passthrough")
    return {
        "logistic_selected": original["logistic_selected"],
        "random_forest": original["random_forest"],
        "rf_without_selection": no_selection,
        "rf_missing_indicator": original["random_forest_missing_indicator"],
    }


def score_metrics(y, probability):
    truth = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    both = len(np.unique(truth)) == 2
    return {
        "rows": len(truth), "failures": int(truth.sum()),
        "failure_rate": float(truth.mean()),
        "average_precision": float(average_precision_score(truth, p)) if both else np.nan,
        "roc_auc": float(roc_auc_score(truth, p)) if both else np.nan,
        "brier_score": float(brier_score_loss(truth, p)),
        "status": "ok" if both else "single_class_ranking_metrics_undefined",
    }


def choose_threshold(y, probability, fn_cost=10, fp_cost=1):
    """Use earlier validation predictions only; include the review-none policy."""
    truth, p = np.asarray(y), np.asarray(probability)
    grid = np.r_[0.0, np.linspace(0.01, 0.99, 99), 1.0, np.nextafter(1.0, 2.0)]
    values = []
    for threshold in grid:
        alert = p >= threshold
        fn, fp = int(((truth == 1) & ~alert).sum()), int(((truth == 0) & alert).sum())
        values.append((fn_cost * fn + fp_cost * fp, fn, fp, -threshold, threshold))
    return float(min(values)[-1])


def fit_policy(X_train, y_train, time_train, config):
    """All selection and threshold fitting are confined to the supplied past."""
    models = study_models(config["seed"])
    split = windows(time_train, config["inner_train_fractions"], config["inner_end_fractions"])
    rows, predictions = [], {}
    for name in config["candidates"]:
        pieces = []
        for fold, (start, end) in enumerate(split, 1):
            if y_train.iloc[:start].nunique() < 2:
                raise ValueError("Earlier training window lacks a class; do not silently omit it.")
            fitted = clone(models[name]).fit(X_train.iloc[:start], y_train.iloc[:start])
            prob = fitted.predict_proba(X_train.iloc[start:end])[:, 1]
            rows.append({"model": name, "inner_window": fold, "train_rows": start,
                         **score_metrics(y_train.iloc[start:end], prob)})
            pieces.append(pd.DataFrame({"position": np.arange(start, end),
                                        "truth": y_train.iloc[start:end].to_numpy(), "risk": prob}))
        predictions[name] = pd.concat(pieces, ignore_index=True)
    comparison = pd.DataFrame(rows)
    means = comparison.groupby("model")["average_precision"].mean()
    if means.isna().all():
        raise ValueError("All inner ranking metrics undefined; cannot choose a model.")
    # Exact score ties use declared candidate order. No post-hoc tuning on outer periods.
    selected = max(config["candidates"], key=lambda n: means[n] if pd.notna(means[n]) else -np.inf)
    earlier = predictions[selected]
    threshold = choose_threshold(earlier.truth, earlier.risk,
                                 config["false_negative_cost"], config["false_positive_cost"])
    fitted = clone(models[selected]).fit(X_train, y_train)
    return fitted, selected, threshold, comparison, earlier


def capacity_evidence(y, probability, capacities, seed=42):
    """Batch review with seeded label-blind tie breaking and an exact random null.

    The null interval describes random review, NOT uncertainty of model lift.
    Capacity is a retrospective batch budget, not a live arrival-time policy.
    """
    truth, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    tie = np.random.default_rng(seed).random(len(truth))
    order = np.lexsort((tie, -p))
    total, n = int(truth.sum()), len(truth)
    rows = []
    for cap in capacities:
        if not 0 < cap <= 1:
            raise ValueError("Review capacity must be in (0, 1].")
        k = max(1, int(np.ceil(n * cap)))
        captured = int(truth[order[:k]].sum())
        rows.append({
            "target_capacity": cap, "reviewed_runs": k, "actual_capacity": k / n,
            "failures": total, "captured": captured, "false_alerts": k - captured,
            "recall": captured / total if total else np.nan, "precision": captured / k,
            "lift": (captured / total) / (k / n) if total else np.nan,
            "random_expected_captured": k * total / n,
            "random_capture_lower95": float(hypergeom.ppf(.025, n, total, k)),
            "random_capture_upper95": float(hypergeom.ppf(.975, n, total, k)),
            "random_p_ge_observed_descriptive": float(hypergeom.sf(captured - 1, n, total, k)),
            "boundary_tie_size": int((p == p[order[k - 1]]).sum()),
        })
    return pd.DataFrame(rows)


def batch_selection(probability, capacity, seed=42):
    p = np.asarray(probability)
    order = np.lexsort((np.random.default_rng(seed).random(len(p)), -p))
    selected = np.zeros(len(p), dtype=bool)
    selected[order[:max(1, int(np.ceil(len(p) * capacity)))]] = True
    return selected


def reference_percentile(reference_scores, scores):
    """Frozen empirical reference scale: appending later runs cannot revise history."""
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("A finite nonempty reference is required.")
    return np.searchsorted(reference, np.asarray(scores), side="right") / len(reference)
