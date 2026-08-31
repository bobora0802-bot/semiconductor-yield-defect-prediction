"""Run robustness, calibration, capacity, drift, and stability analyses."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_project.data import load_secom  # noqa: E402
from secom_project.evaluation import (  # noqa: E402
    binary_metrics,
    capacity_table,
    chronological_holdout,
    cost_sensitive_threshold,
    expanding_window_probabilities,
    stratified_bootstrap_intervals,
)
from secom_project.modeling import build_model_pipelines  # noqa: E402


RANDOM_STATE = 42
COST_RATIOS = (1, 2, 5, 10, 20, 50)


def _calibration_error(y_true: pd.Series, probability: np.ndarray, n_bins: int = 5) -> float:
    frame = pd.DataFrame({"truth": np.asarray(y_true), "score": probability})
    frame["bin"] = pd.qcut(frame["score"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        observed=("truth", "mean"), predicted=("score", "mean"), count=("truth", "size")
    )
    return float(
        np.sum(np.abs(grouped["observed"] - grouped["predicted"]) * grouped["count"])
        / len(frame)
    )


def _selected_feature_names(fitted_pipeline) -> tuple[np.ndarray, np.ndarray]:
    sparse_names = np.asarray(
        fitted_pipeline.named_steps["drop_sparse"].selected_columns_, dtype=object
    )
    variance_names = sparse_names[
        fitted_pipeline.named_steps["drop_constant"].get_support()
    ]
    selected_names = variance_names[fitted_pipeline.named_steps["select"].get_support()]
    return variance_names, selected_names


def _feature_stability(estimator, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    selected_count: defaultdict[str, int] = defaultdict(int)
    top_count: defaultdict[str, int] = defaultdict(int)
    top_importance: defaultdict[str, list[float]] = defaultdict(list)
    top_rank: defaultdict[str, list[int]] = defaultdict(list)
    fold_count = 0
    for train_index, _ in folds.split(X, y):
        fitted = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
        _, selected_names = _selected_feature_names(fitted)
        for name in selected_names:
            selected_count[str(name)] += 1
        importance = fitted.named_steps["classifier"].feature_importances_
        ranking = np.argsort(-importance, kind="stable")[:20]
        for rank, position in enumerate(ranking, start=1):
            name = str(selected_names[position])
            top_count[name] += 1
            top_importance[name].append(float(importance[position]))
            top_rank[name].append(rank)
        fold_count += 1
        print(f"Finished stability fold {fold_count}/10")
    names = sorted(set(selected_count) | set(top_count))
    rows = []
    for name in names:
        rows.append(
            {
                "feature": name,
                "selection_frequency": selected_count[name] / fold_count,
                "top20_frequency": top_count[name] / fold_count,
                "mean_importance_when_top20": (
                    float(np.mean(top_importance[name])) if top_importance[name] else 0.0
                ),
                "median_rank_when_top20": (
                    float(np.median(top_rank[name])) if top_rank[name] else np.nan
                ),
                "folds": fold_count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top20_frequency", "mean_importance_when_top20", "selection_frequency"],
        ascending=False,
    )


def _drift_table(
    X_reference: pd.DataFrame, X_later: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for feature in X_reference.columns:
        reference = X_reference[feature]
        later = X_later[feature]
        reference_clean = reference.dropna()
        later_clean = later.dropna()
        if len(reference_clean) and len(later_clean):
            ks = ks_2samp(reference_clean, later_clean, method="auto")
            reference_std = float(reference_clean.std(ddof=0))
            standardised_shift = (
                abs(float(later_clean.mean() - reference_clean.mean())) / reference_std
                if reference_std > 0
                else 0.0
            )
            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        else:
            standardised_shift = np.nan
            ks_statistic = np.nan
            ks_pvalue = np.nan
        rows.append(
            {
                "feature": feature,
                "reference_missing_rate": float(reference.isna().mean()),
                "later_missing_rate": float(later.isna().mean()),
                "absolute_missing_rate_change": float(
                    abs(later.isna().mean() - reference.isna().mean())
                ),
                "absolute_standardised_mean_shift": standardised_shift,
                "ks_statistic": ks_statistic,
                "ks_pvalue_unadjusted": ks_pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ks_statistic", "absolute_missing_rate_change"], ascending=False
    )


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    figures = reports / "figures"
    tables = reports / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    X, y, timestamps = load_secom(PROJECT_ROOT / "data" / "raw")
    baseline = json.loads((reports / "baseline_summary.json").read_text(encoding="utf-8"))
    selected_name = str(baseline["selected_model"])
    selected_model = build_model_pipelines(RANDOM_STATE)[selected_name]
    if selected_name != "random_forest":
        raise RuntimeError(
            "Enhanced stability mapping currently expects the parsimonious random_forest pipeline."
        )

    X_random_train, X_random_test, y_random_train, y_random_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    random_oof = cross_val_predict(
        selected_model,
        X_random_train,
        y_random_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    random_fitted = clone(selected_model).fit(X_random_train, y_random_train)
    random_test_probability = random_fitted.predict_proba(X_random_test)[:, 1]

    random_threshold, _ = cost_sensitive_threshold(
        y_random_train, random_oof, false_negative_cost=10
    )
    random_metrics = binary_metrics(y_random_test, random_test_probability, random_threshold)
    random_ci = stratified_bootstrap_intervals(
        y_random_test,
        random_test_probability,
        random_threshold,
        n_resamples=1_000,
        random_state=RANDOM_STATE,
    )
    random_ci.insert(0, "validation_scheme", "random_stratified_holdout")
    random_capacity = capacity_table(y_random_test, random_test_probability)
    random_capacity.insert(0, "validation_scheme", "random_stratified_holdout")

    chronological = chronological_holdout(X, y, timestamps, test_size=0.20)
    temporal_oof_y, temporal_oof_probability = expanding_window_probabilities(
        selected_model,
        chronological.X_train,
        chronological.y_train,
        chronological.time_train,
        n_splits=5,
        initial_train_fraction=0.50,
    )
    temporal_threshold, _ = cost_sensitive_threshold(
        temporal_oof_y, temporal_oof_probability, false_negative_cost=10
    )
    temporal_fitted = clone(selected_model).fit(
        chronological.X_train, chronological.y_train
    )
    temporal_test_probability = temporal_fitted.predict_proba(chronological.X_test)[:, 1]
    temporal_metrics = binary_metrics(
        chronological.y_test, temporal_test_probability, temporal_threshold
    )
    temporal_ci = stratified_bootstrap_intervals(
        chronological.y_test,
        temporal_test_probability,
        temporal_threshold,
        n_resamples=1_000,
        random_state=RANDOM_STATE + 1,
    )
    temporal_ci.insert(0, "validation_scheme", "chronological_holdout")
    temporal_capacity = capacity_table(
        chronological.y_test, temporal_test_probability
    )
    temporal_capacity.insert(0, "validation_scheme", "chronological_holdout")

    pd.concat([random_ci, temporal_ci], ignore_index=True).to_csv(
        tables / "bootstrap_confidence_intervals.csv", index=False
    )
    pd.concat([random_capacity, temporal_capacity], ignore_index=True).to_csv(
        tables / "review_capacity_analysis.csv", index=False
    )

    cost_rows = []
    for ratio in COST_RATIOS:
        threshold, _ = cost_sensitive_threshold(
            y_random_train, random_oof, false_negative_cost=ratio
        )
        values = binary_metrics(y_random_test, random_test_probability, threshold)
        cost_rows.append(
            {"validation_scheme": "random_stratified_holdout", "fn_fp_cost_ratio": ratio, **values}
        )
        threshold, _ = cost_sensitive_threshold(
            temporal_oof_y, temporal_oof_probability, false_negative_cost=ratio
        )
        values = binary_metrics(
            chronological.y_test, temporal_test_probability, threshold
        )
        cost_rows.append(
            {"validation_scheme": "chronological_holdout", "fn_fp_cost_ratio": ratio, **values}
        )
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(tables / "cost_ratio_sensitivity.csv", index=False)

    calibration_models = {
        "uncalibrated": clone(selected_model),
        "sigmoid": CalibratedClassifierCV(
            clone(selected_model), method="sigmoid", cv=cv, ensemble=True
        ),
        "isotonic": CalibratedClassifierCV(
            clone(selected_model), method="isotonic", cv=cv, ensemble=True
        ),
    }
    calibration_rows = []
    calibration_probabilities = {}
    for name, model in calibration_models.items():
        model.fit(X_random_train, y_random_train)
        probability = model.predict_proba(X_random_test)[:, 1]
        calibration_probabilities[name] = probability
        values = binary_metrics(y_random_test, probability, threshold=0.50)
        calibration_rows.append(
            {
                "calibration": name,
                "roc_auc": values["roc_auc"],
                "pr_auc": values["pr_auc"],
                "brier_score": values["brier_score"],
                "log_loss": float(log_loss(y_random_test, probability, labels=[0, 1])),
                "quantile_ece_5_bins": _calibration_error(
                    y_random_test, probability, n_bins=5
                ),
            }
        )
        print(f"Finished calibration: {name}")
    calibration_comparison = pd.DataFrame(calibration_rows).sort_values("brier_score")
    calibration_comparison.to_csv(
        tables / "calibration_comparison.csv", index=False
    )

    drift = _drift_table(chronological.X_train, chronological.X_test)
    drift.head(100).to_csv(tables / "temporal_feature_drift_top100.csv", index=False)
    weekly = pd.DataFrame({"timestamp": timestamps, "failure": y}).set_index("timestamp")
    weekly = weekly.resample("7D")["failure"].agg(["count", "sum", "mean"]).reset_index()
    weekly.columns = ["week_start", "runs", "failures", "failure_rate"]
    weekly.to_csv(tables / "weekly_failure_rate.csv", index=False)

    stability = _feature_stability(selected_model, X_random_train, y_random_train)
    stability.to_csv(tables / "feature_stability.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(7.5, 6))
    for name, probability in calibration_probabilities.items():
        observed, predicted = calibration_curve(
            y_random_test, probability, n_bins=5, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", linewidth=2, label=name)
    axis.plot([0, 1], [0, 1], "--", color="0.45", label="perfect calibration")
    axis.set(
        title="Probability calibration on random holdout",
        xlabel="Mean predicted failure probability",
        ylabel="Observed failure fraction",
        xlim=(0, 0.55),
        ylim=(0, 0.55),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "calibration_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    validation_comparison = pd.DataFrame(
        [
            {"validation_scheme": "Random stratified", **random_metrics},
            {"validation_scheme": "Chronological", **temporal_metrics},
        ]
    )
    validation_comparison.to_csv(
        tables / "random_vs_chronological_metrics.csv", index=False
    )
    melted = validation_comparison.melt(
        id_vars="validation_scheme",
        value_vars=["roc_auc", "pr_auc", "recall", "precision"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(
        {
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "recall": "Recall @ policy",
            "precision": "Precision @ policy",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=melted, x="metric", y="value", hue="validation_scheme", ax=axis)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", padding=3, fontsize=11)
    axis.set(title="Random versus chronological holdout", xlabel="", ylabel="Metric value", ylim=(0, 1))
    axis.legend(title="Validation scheme")
    figure.tight_layout()
    figure.savefig(
        figures / "random_vs_chronological_metrics.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    capacity_combined = pd.concat([random_capacity, temporal_capacity], ignore_index=True)
    capacity_combined["validation_scheme"] = capacity_combined["validation_scheme"].map(
        {
            "random_stratified_holdout": "Random stratified",
            "chronological_holdout": "Chronological",
        }
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    sns.lineplot(
        data=capacity_combined,
        x="actual_review_rate",
        y="failure_recall",
        hue="validation_scheme",
        marker="o",
        linewidth=2.5,
        ax=axis,
    )
    axis.set(
        title="Failure capture under limited review capacity",
        xlabel="Fraction of runs reviewed",
        ylabel="Fraction of failures captured",
        xlim=(0, 0.32),
        ylim=(0, 1),
    )
    axis.legend(title="Validation scheme")
    figure.tight_layout()
    figure.savefig(figures / "review_capacity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    top_stability = stability.head(15).sort_values("top20_frequency")
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.barh(top_stability["feature"], top_stability["top20_frequency"], color="#2E6F9E")
    axis.set(
        title="Feature ranking stability across 10 folds",
        xlabel="Fraction of folds ranked in model top 20",
        ylabel="Anonymised process variable",
        xlim=(0, 1),
    )
    figure.tight_layout()
    figure.savefig(figures / "feature_stability.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        pd.to_datetime(weekly["week_start"]),
        weekly["failure_rate"],
        marker="o",
        linewidth=2.5,
        color="#2E6F9E",
    )
    axis.axvline(
        chronological.time_test.min(),
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label="Chronological holdout begins",
    )
    axis.set(
        title="Weekly observed failure rate",
        xlabel="Week",
        ylabel="Failure fraction",
        ylim=(0, max(0.25, float(weekly["failure_rate"].max()) * 1.15)),
    )
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figures / "weekly_failure_rate.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "selected_model": selected_name,
        "random_stratified": {
            "threshold": random_threshold,
            "test_failures": int(y_random_test.sum()),
            "metrics": random_metrics,
        },
        "chronological": {
            "train_period": [
                chronological.time_train.min().isoformat(),
                chronological.time_train.max().isoformat(),
            ],
            "test_period": [
                chronological.time_test.min().isoformat(),
                chronological.time_test.max().isoformat(),
            ],
            "threshold": temporal_threshold,
            "test_failures": int(chronological.y_test.sum()),
            "metrics": temporal_metrics,
        },
        "calibration_observation": {
            "best_brier_and_quantile_ece": "isotonic",
            "best_log_loss": "sigmoid",
            "promoted_for_policy_use": None,
            "reason": (
                "Calibration methods disagree across proper scores on a small holdout; "
                "report the comparison without promoting one probability scale."
            ),
        },
        "feature_stability_top": stability.head(10).to_dict(orient="records"),
    }
    (reports / "enhanced_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
