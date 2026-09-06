"""Run leakage-safe univariate and multivariate Phase I/II monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_project.data import load_secom  # noqa: E402
from secom_project.modeling import build_model_pipelines  # noqa: E402
from secom_project.spc import (  # noqa: E402
    MultivariateSPC,
    UnivariateSPC,
    alert_metrics,
    define_phases,
    select_monitoring_features,
)

RANDOM_STATE = 42
CAPACITIES = (0.05, 0.10, 0.20, 0.30)


def _capacity_rows(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    validation_scores: dict[str, np.ndarray],
    validation_truth: np.ndarray,
) -> pd.DataFrame:
    rows = []
    selections: dict[tuple[str, float], set[int]] = {}
    for name, score in scores.items():
        order = np.argsort(-score, kind="stable")
        val_order = np.argsort(-validation_scores[name], kind="stable")
        for capacity in CAPACITIES:
            count = max(1, int(np.ceil(len(truth) * capacity)))
            val_count = max(1, int(np.ceil(len(validation_truth) * capacity)))
            selected = set(order[:count].tolist())
            selections[(name, capacity)] = selected
            captured = int(truth[list(selected)].sum())
            val_captured = int(validation_truth[val_order[:val_count]].sum())
            recall = captured / max(int(truth.sum()), 1)
            val_recall = val_captured / max(int(validation_truth.sum()), 1)
            precision = captured / count
            actual_review_rate = count / len(truth)
            rows.append(
                {
                    "strategy": name,
                    "review_capacity": capacity,
                    "reviewed_runs": count,
                    "actual_review_rate": actual_review_rate,
                    "failures_captured": captured,
                    "failure_capture": recall,
                    "false_alerts": count - captured,
                    "review_precision": precision,
                    "lift_over_random": recall / actual_review_rate,
                    "development_failure_capture": val_recall,
                    "chronological_robustness_gap": abs(val_recall - recall),
                }
            )
    table = pd.DataFrame(rows)
    overlaps = []
    for row in table.itertuples():
        selected = selections[(row.strategy, row.review_capacity)]
        ml = selections[("existing_ml_risk", row.review_capacity)]
        overlaps.append(len(selected & ml) / max(len(selected | ml), 1))
    table["jaccard_overlap_with_ml"] = overlaps
    return table


def _plot_timeline(frame: pd.DataFrame, score: str, limit: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(frame["timestamp"], frame[score], linewidth=1.1, color="#2E6F9E")
    ax.axhline(limit, color="#C44E52", linestyle="--", label="Phase I limit")
    for phase, color in (("phase_ii", "#DD8452"), ("final_holdout", "#C44E52")):
        start = frame.loc[frame["phase"] == phase, "timestamp"].min()
        ax.axvline(start, color=color, linestyle=":" if phase == "phase_ii" else "--", label=f"{phase} begins")
    failures = frame[frame["failure"] == 1]
    ax.scatter(failures["timestamp"], failures[score], marker="x", color="black", s=35, label="observed failure (offline)")
    ax.set(title=title, xlabel="Run timestamp", ylabel=score.upper())
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    X, y, timestamps = load_secom(PROJECT_ROOT / "data" / "raw")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    holdout_rows = int(np.ceil(len(X) * 0.20))
    train_size = len(X) - holdout_rows
    phases = define_phases(y, train_size=train_size, phase_i_fraction=0.60)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    features = select_monitoring_features(X_train, y_train, top_n=15)
    reference = X.iloc[phases.reference_positions][features]

    univariate = UnivariateSPC().fit(reference)
    univariate_scores = univariate.transform(X[features]).reset_index(drop=True)
    mspc = MultivariateSPC().fit(reference)
    mspc_scores = mspc.transform(X[features]).reset_index(drop=True)
    residual_contributions = mspc.q_contributions(X[features]).reset_index(drop=True)

    baseline = json.loads((reports / "baseline_summary.json").read_text(encoding="utf-8"))
    model_name = str(baseline["selected_model"])
    model = build_model_pipelines(RANDOM_STATE)[model_name]
    model.fit(X_train, y_train)
    model_risk = model.predict_proba(X)[:, 1]

    frame = pd.DataFrame(
        {
            "run_id": np.arange(len(X)),
            "timestamp": timestamps,
            "failure": y,
            "phase": phases.phase_labels,
            "model_risk": model_risk,
        }
    )
    keep_univariate = [
        column for column in univariate_scores
        if column.endswith(("__z", "__ewma", "__cusum", "__imr_alert", "__ewma_alert", "__cusum_alert"))
        or column in {"imr_alert", "ewma_alert", "cusum_alert", "ewma_score", "cusum_score"}
    ]
    frame = pd.concat([frame, univariate_scores[keep_univariate], mspc_scores.drop(columns="_residual_contributions")], axis=1)
    frame["spc_alert"] = frame[["imr_alert", "ewma_alert", "cusum_alert", "mspc_alert"]].any(axis=1)
    frame["spc_score"] = frame[["ewma_score", "cusum_score", "t2_ratio", "q_ratio"]].max(axis=1)
    frame.to_csv(tables / "mspc_run_scores.csv", index=False)
    residual_contributions.insert(0, "run_id", np.arange(len(X)))
    residual_contributions.to_csv(tables / "mspc_q_contributions.csv", index=False)

    holdout = frame.iloc[train_size:].reset_index(drop=True)
    method_rows = []
    for name, alert_column in (("I-MR", "imr_alert"), ("EWMA", "ewma_alert"), ("CUSUM", "cusum_alert"), ("Hotelling T2", "t2_alert"), ("Q/SPE", "q_alert"), ("MSPC union", "mspc_alert"), ("All SPC union", "spc_alert")):
        method_rows.append({"method": name, **alert_metrics(holdout["failure"], holdout[alert_column])})
    method_table = pd.DataFrame(method_rows)
    method_table.to_csv(tables / "spc_method_comparison.csv", index=False)

    validation_start = int(np.floor(train_size * 0.80))
    validation = frame.iloc[validation_start:train_size].reset_index(drop=True)
    development_model = clone(model).fit(X.iloc[:validation_start], y.iloc[:validation_start])
    development_model_risk = development_model.predict_proba(X.iloc[validation_start:train_size])[:, 1]
    validation_truth = validation["failure"].to_numpy(dtype=int)
    holdout_truth = holdout["failure"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    random_holdout = rng.random(len(holdout))
    random_validation = rng.random(len(validation))
    best_weight = 0.5
    best_capture = -1.0
    for weight in np.linspace(0, 1, 5):
        combined = weight * pd.Series(development_model_risk).rank(pct=True).to_numpy() + (1 - weight) * validation["spc_score"].rank(pct=True).to_numpy()
        capture = np.mean([
            validation_truth[np.argsort(-combined, kind="stable")[: max(1, int(np.ceil(len(validation) * cap)))]].sum() / max(validation_truth.sum(), 1)
            for cap in CAPACITIES
        ])
        if capture > best_capture:
            best_capture, best_weight = float(capture), float(weight)
    holdout_combined = best_weight * holdout["model_risk"].rank(pct=True).to_numpy() + (1 - best_weight) * holdout["spc_score"].rank(pct=True).to_numpy()
    validation_combined = best_weight * pd.Series(development_model_risk).rank(pct=True).to_numpy() + (1 - best_weight) * validation["spc_score"].rank(pct=True).to_numpy()
    strategies = {
        "random_review": random_holdout,
        "existing_ml_risk": holdout["model_risk"].to_numpy(),
        "spc_alert_ranking": holdout["spc_score"].to_numpy(),
        "ml_spc_combined": holdout_combined,
    }
    validation_strategies = {
        "random_review": random_validation,
        "existing_ml_risk": development_model_risk,
        "spc_alert_ranking": validation["spc_score"].to_numpy(),
        "ml_spc_combined": validation_combined,
    }
    strategy_table = _capacity_rows(holdout_truth, strategies, validation_strategies, validation_truth)
    strategy_table["combined_ml_weight"] = np.where(strategy_table["strategy"].eq("ml_spc_combined"), best_weight, np.nan)
    strategy_table.to_csv(tables / "monitoring_strategy_comparison.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    _plot_timeline(frame, "t2", mspc.t2_limit_, "Hotelling's T² across Phase I, Phase II and final holdout", figures / "mspc_t2_timeline.png")
    _plot_timeline(frame, "q_spe", mspc.q_limit_, "PCA Q/SPE across Phase I, Phase II and final holdout", figures / "mspc_q_timeline.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=method_table, x="method", y="failure_recall", hue="method", legend=False, ax=ax)
    ax.set(title="Final-holdout SPC alert comparison", xlabel="", ylabel="Failure capture")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figures / "spc_alert_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(data=strategy_table, x="review_capacity", y="failure_capture", hue="strategy", marker="o", ax=ax)
    ax.plot(CAPACITIES, CAPACITIES, "--", color="0.4", label="random expectation")
    ax.set(title="Monitoring strategy under review-capacity constraints", xlabel="Review capacity", ylabel="Final-holdout failure capture", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "monitoring_strategy_capacity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scope": "Semiconductor Yield Excursion Monitoring & Root-Cause Triage",
        "reference_policy": "label-assisted historical reference: known pass runs in the earliest 60% of chronological training; not a validated in-control production period",
        "split": {
            "phase_i_window_rows": phases.phase_i_size,
            "phase_i_reference_pass_rows": len(phases.reference_positions),
            "phase_ii_rows": train_size - phases.phase_i_size,
            "final_holdout_rows": holdout_rows,
            "final_holdout_start": timestamps.iloc[train_size].isoformat(),
        },
        "monitoring_features": features,
        "mspc": {
            "pca_components": mspc.n_components_,
            "variance_target": mspc.variance_target,
            "threshold_method": mspc.threshold_method_,
            "t2_limit": mspc.t2_limit_,
            "q_spe_limit": mspc.q_limit_,
        },
        "combined_score": {
            "ml_weight": best_weight,
            "spc_weight": 1 - best_weight,
            "selection_source": "earlier chronological development validation only",
        },
        "spc_final_holdout": method_table.to_dict(orient="records"),
        "strategy_comparison": strategy_table.astype(object).where(pd.notna(strategy_table), None).to_dict(orient="records"),
        "interpretation": "SPC alerts are process-excursion triage signals; they are not supervised failure classifications and do not establish causality.",
    }
    (reports / "spc_triage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
