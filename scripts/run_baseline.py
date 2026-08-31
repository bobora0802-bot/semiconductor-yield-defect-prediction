"""Train, select, and evaluate leakage-aware SECOM failure classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_project.data import load_secom  # noqa: E402
from secom_project.modeling import build_model_pipelines  # noqa: E402


RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_counts(y_true, prediction)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _confusion_counts(y_true: pd.Series, prediction: np.ndarray) -> tuple[int, int, int, int]:
    truth = np.asarray(y_true, dtype=int)
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    return tn, fp, fn, tp


def _cost_sensitive_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = _confusion_counts(y_true, prediction)
        rows.append(
            {
                "threshold": float(threshold),
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "true_negative": tn,
                "expected_cost": float(FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn),
                "recall": float(recall_score(y_true, prediction, zero_division=0)),
                "precision": float(precision_score(y_true, prediction, zero_division=0)),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["expected_cost", "false_negative", "false_positive", "threshold"],
        ascending=[True, True, True, False],
    ).iloc[0]
    return float(best["threshold"]), table


def _plot_curves(y_true: pd.Series, probability: np.ndarray, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    fpr, tpr, _ = roc_curve(y_true, probability)
    axes[0].plot(fpr, tpr, linewidth=2.5, label=f"AUC = {roc_auc_score(y_true, probability):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
    axes[0].set(title="Held-out ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, probability)
    baseline = float(np.mean(y_true))
    axes[1].plot(recall, precision, linewidth=2.5, label=f"AP = {average_precision_score(y_true, probability):.3f}")
    axes[1].axhline(baseline, linestyle="--", color="0.55", linewidth=1, label=f"Failure rate = {baseline:.3f}")
    axes[1].set(title="Held-out precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(figures_dir / "heldout_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X, y, timestamps = load_secom(PROJECT_ROOT / "data" / "raw")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    profile = pd.DataFrame(
        [
            {
                "samples": len(X),
                "features": X.shape[1],
                "failures": int(y.sum()),
                "failure_rate": float(y.mean()),
                "missing_cells": int(X.isna().sum().sum()),
                "missing_fraction": float(X.isna().mean().mean()),
                "features_over_50pct_missing": int((X.isna().mean() > 0.50).sum()),
                "start_timestamp": timestamps.min().isoformat(),
                "end_timestamp": timestamps.max().isoformat(),
            }
        ]
    )
    profile.to_csv(tables_dir / "dataset_profile.csv", index=False)

    candidates = build_model_pipelines(random_state=RANDOM_STATE)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
    }
    comparison_rows: list[dict[str, float | str]] = []
    for name, pipeline in candidates.items():
        result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
        comparison_rows.append(row)
        print(f"Finished CV: {name}")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_pr_auc_mean", "cv_recall_mean"], ascending=False
    )
    comparison.to_csv(tables_dir / "model_cv_comparison.csv", index=False)
    best_pr_auc = float(comparison.iloc[0]["cv_pr_auc_mean"])
    practical_ties = comparison[
        comparison["cv_pr_auc_mean"] >= best_pr_auc - 0.005
    ].sort_values(
        ["cv_pr_auc_std", "cv_roc_auc_mean"], ascending=[True, False]
    )
    selected_name = str(practical_ties.iloc[0]["model"])
    selected_model = candidates[selected_name]

    threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    oof_probability = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        cv=threshold_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    selected_threshold, threshold_table = _cost_sensitive_threshold(y_train, oof_probability)
    threshold_table.to_csv(tables_dir / "threshold_analysis.csv", index=False)

    selected_model.fit(X_train, y_train)
    test_probability = selected_model.predict_proba(X_test)[:, 1]
    default_metrics = _metrics(y_test, test_probability, threshold=0.50)
    selected_metrics = _metrics(y_test, test_probability, threshold=selected_threshold)

    summary = {
        "selected_model": selected_name,
        "selection_metric": "mean repeated-CV PR-AUC on the training partition",
        "selection_rule": (
            "Models within 0.005 absolute PR-AUC of the best mean are treated as "
            "practical ties; select the candidate with lower PR-AUC variability, "
            "then higher ROC-AUC."
        ),
        "threshold_policy": {
            "source": "out-of-fold probabilities from the training partition only",
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "selected_threshold": selected_threshold,
        },
        "holdout_size": len(X_test),
        "holdout_failures": int(y_test.sum()),
        "default_threshold_metrics": default_metrics,
        "cost_sensitive_metrics": selected_metrics,
    }
    (reports_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _plot_curves(y_test, test_probability, figures_dir)
    prediction = (test_probability >= selected_threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction,
        display_labels=["Pass", "Fail"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Held-out confusion matrix (threshold={selected_threshold:.2f})")
    display.figure_.tight_layout()
    display.figure_.savefig(
        figures_dir / "heldout_confusion_matrix.png", dpi=180, bbox_inches="tight"
    )
    plt.close(display.figure_)

    importance = permutation_importance(
        selected_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": X.columns,
            "permutation_importance_mean": importance.importances_mean,
            "permutation_importance_std": importance.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    importance_table.head(30).to_csv(
        tables_dir / "top_permutation_importance.csv", index=False
    )

    top = importance_table.head(15).sort_values("permutation_importance_mean")
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.barh(
        top["feature"],
        top["permutation_importance_mean"],
        xerr=top["permutation_importance_std"],
        color="#2E6F9E",
        alpha=0.9,
    )
    axis.set(
        title="Held-out permutation importance",
        xlabel="Decrease in average precision after permutation",
        ylabel="Anonymised process variable",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "top_permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
