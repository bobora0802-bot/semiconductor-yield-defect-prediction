"""Build association-based candidate process-variable and run triage outputs."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.inspection import permutation_importance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_project.data import load_secom  # noqa: E402
from secom_project.modeling import build_model_pipelines  # noqa: E402
from secom_project.triage import build_consensus_ranking, make_run_level_triage  # noqa: E402

RANDOM_STATE = 42


def _pipeline_names(fitted) -> tuple[np.ndarray, np.ndarray]:
    sparse = np.asarray(fitted.named_steps["drop_sparse"].selected_columns_, dtype=object)
    variance = sparse[fitted.named_steps["drop_constant"].get_support()]
    selected = variance[fitted.named_steps["select"].get_support()]
    return variance, selected


def _training_stability(model, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    selected_count: defaultdict[str, int] = defaultdict(int)
    top_count: defaultdict[str, int] = defaultdict(int)
    ranks: defaultdict[str, list[int]] = defaultdict(list)
    ends = [int(len(X_train) * fraction) for fraction in (0.55, 0.70, 0.85, 1.0)]
    for end in ends:
        fitted = clone(model).fit(X_train.iloc[:end], y_train.iloc[:end])
        _, selected = _pipeline_names(fitted)
        for feature in selected:
            selected_count[str(feature)] += 1
        order = np.argsort(-fitted.named_steps["classifier"].feature_importances_, kind="stable")
        for rank, position in enumerate(order, start=1):
            feature = str(selected[position])
            ranks[feature].append(rank)
            if rank <= 20:
                top_count[feature] += 1
    rows = []
    for feature in X_train.columns:
        observed = ranks[str(feature)]
        stability = 1 / (1 + float(np.std(observed))) if observed else 0.0
        rows.append(
            {
                "feature": feature,
                "selection_frequency": selected_count[str(feature)] / len(ends),
                "top20_frequency": top_count[str(feature)] / len(ends),
                "rank_stability": stability,
            }
        )
    return pd.DataFrame(rows)


def _effect_size(pass_values: pd.Series, failure_values: pd.Series) -> float:
    a, b = pass_values.dropna().to_numpy(), failure_values.dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else 0.0


def main() -> None:
    tables = PROJECT_ROOT / "reports" / "tables"
    figures = PROJECT_ROOT / "reports" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    run_scores_path = tables / "mspc_run_scores.csv"
    if not run_scores_path.exists():
        raise FileNotFoundError("Run scripts/run_spc_analysis.py first.")

    X, y, timestamps = load_secom(PROJECT_ROOT / "data" / "raw")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    run_scores = pd.read_csv(run_scores_path, parse_dates=["timestamp"])
    contributions = pd.read_csv(tables / "mspc_q_contributions.csv")
    train_size = int((run_scores["phase"] != "final_holdout").sum())
    phase_i_size = int((run_scores["phase"] == "phase_i").sum())
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_phase_i = X.iloc[:phase_i_size]
    X_phase_ii = X.iloc[phase_i_size:train_size]

    model = build_model_pipelines(RANDOM_STATE)["random_forest"]
    stability = _training_stability(model, X_train, y_train)
    validation_start = int(np.floor(train_size * 0.80))
    fitted = clone(model).fit(X_train.iloc[:validation_start], y_train.iloc[:validation_start])
    perm = permutation_importance(
        fitted,
        X_train.iloc[validation_start:],
        y_train.iloc[validation_start:],
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    permutation = dict(zip(X.columns, perm.importances_mean, strict=True))

    selected_features = [column.removesuffix("__z") for column in run_scores.columns if column.endswith("__z")]
    q_mean = contributions.loc[phase_i_size:train_size - 1, selected_features].mean().to_dict()
    rows = []
    for feature in X.columns:
        effect = _effect_size(X_train.loc[y_train == 0, feature], X_train.loc[y_train == 1, feature])
        phase_i_clean = X_phase_i[feature].dropna()
        phase_ii_clean = X_phase_ii[feature].dropna()
        ks = float(ks_2samp(phase_i_clean, phase_ii_clean).statistic) if len(phase_i_clean) and len(phase_ii_clean) else 0.0
        if feature in selected_features:
            alerts = run_scores.loc[
                :train_size - 1,
                [f"{feature}__ewma_alert", f"{feature}__cusum_alert"],
            ]
            spc_rate = float(alerts.any(axis=1).mean())
        else:
            spc_rate = 0.0
        rows.append(
            {
                "feature": feature,
                "permutation_importance": float(permutation[feature]),
                "effect_size": effect,
                "absolute_effect_size": abs(effect),
                "missing_rate_change": float(abs(X_phase_ii[feature].isna().mean() - X_phase_i[feature].isna().mean())),
                "ks_statistic": ks,
                "spc_alert_rate": spc_rate,
                "q_contribution": float(q_mean.get(feature, 0.0)),
            }
        )
    evidence = pd.DataFrame(rows).merge(stability, on="feature", how="left")
    candidates = build_consensus_ranking(evidence)
    required_order = [
        "feature", "selection_frequency", "top20_frequency", "permutation_importance",
        "effect_size", "effect_direction", "missing_rate_change", "ks_statistic",
        "spc_alert_rate", "rank_stability", "consensus_rank", "interpretation",
        "absolute_effect_size", "q_contribution", "consensus_score",
    ]
    candidates[required_order].to_csv(tables / "candidate_process_variables.csv", index=False)

    triage = make_run_level_triage(run_scores, contributions)
    triage.to_csv(tables / "run_level_triage.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    top = candidates.head(20).sort_values("consensus_score")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["consensus_score"], color="#2E6F9E")
    ax.set(title="Candidate process-variable consensus ranking", xlabel="Consensus evidence score", ylabel="Anonymous variable")
    fig.tight_layout()
    fig.savefig(figures / "candidate_variable_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    effects = candidates.reindex(candidates["effect_size"].abs().sort_values(ascending=False).index).head(20).sort_values("effect_size")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.where(effects["effect_size"] >= 0, "#C44E52", "#2E6F9E")
    ax.barh(effects["feature"], effects["effect_size"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Training-only failure vs pass standardized effects", xlabel="Signed standardized effect size", ylabel="Anonymous variable")
    fig.tight_layout()
    fig.savefig(figures / "failure_vs_pass_effect_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    chart_features = [feature for feature in candidates["feature"] if feature in selected_features][:6]
    fig, axes = plt.subplots(len(chart_features), 1, figsize=(12, 2.5 * len(chart_features)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, chart_features, strict=True):
        ax.plot(run_scores["timestamp"], run_scores[f"{feature}__z"], linewidth=0.8)
        ax.axhline(3, color="#C44E52", linestyle="--")
        ax.axhline(-3, color="#C44E52", linestyle="--")
        ax.axvline(run_scores.loc[phase_i_size, "timestamp"], color="#DD8452", linestyle=":")
        ax.axvline(run_scores.loc[train_size, "timestamp"], color="#C44E52", linestyle="--")
        ax.set_ylabel(feature)
    axes[0].set_title("Top candidate variable control-chart signals (Phase I / Phase II / holdout)")
    axes[-1].set_xlabel("Run timestamp")
    fig.tight_layout()
    fig.savefig(figures / "top_sensor_control_charts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(candidates[required_order].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
