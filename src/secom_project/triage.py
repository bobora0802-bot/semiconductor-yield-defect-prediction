"""Candidate-variable consensus ranking and run-level triage helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = (
    "selection_frequency",
    "top20_frequency",
    "permutation_importance",
    "absolute_effect_size",
    "missing_rate_change",
    "ks_statistic",
    "spc_alert_rate",
    "q_contribution",
    "rank_stability",
)


def build_consensus_ranking(evidence: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic, missing-safe percentile-rank consensus."""

    required = {"feature", "effect_size", *EVIDENCE_COLUMNS}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"Missing evidence columns: {sorted(missing)}")
    ranked = evidence.copy()
    numeric = list(EVIDENCE_COLUMNS) + ["effect_size"]
    ranked[numeric] = ranked[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    percentile_columns = []
    for column in EVIDENCE_COLUMNS:
        target = f"_{column}_percentile"
        ranked[target] = ranked[column].rank(method="average", pct=True)
        percentile_columns.append(target)
    ranked["consensus_score"] = ranked[percentile_columns].mean(axis=1)
    ranked = ranked.sort_values(["consensus_score", "feature"], ascending=[False, True]).reset_index(drop=True)
    ranked["consensus_rank"] = np.arange(1, len(ranked) + 1)
    ranked["effect_direction"] = np.select(
        [ranked["effect_size"] > 0, ranked["effect_size"] < 0],
        ["higher among failures", "lower among failures"],
        default="no material signed shift",
    )
    ranked["interpretation"] = (
        "Candidate process variable; association-based triage signal for engineering investigation priority that does not establish causality."
    )
    return ranked.drop(columns=percentile_columns)


def make_run_level_triage(run_scores: pd.DataFrame, contributions: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Create exactly one offline triage record for each process run."""

    if len(run_scores) != len(contributions):
        raise ValueError("Run scores and contributions must have the same row count.")
    feature_columns = [column for column in contributions.columns if column != "run_id"]
    values = contributions[feature_columns].to_numpy(dtype=float)
    top_positions = np.argsort(-values, axis=1, kind="stable")[:, :top_n]
    top_variables = ["; ".join(feature_columns[position] for position in row) for row in top_positions]
    model_pct = run_scores["model_risk"].rank(method="average", pct=True)
    spc_pct = run_scores["spc_score"].rank(method="average", pct=True)
    priority_score = np.maximum(model_pct, spc_pct)
    priority = pd.cut(
        priority_score,
        bins=[-np.inf, 0.70, 0.90, np.inf],
        labels=["routine", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return pd.DataFrame(
        {
            "run_id": run_scores["run_id"].to_numpy(),
            "timestamp": run_scores["timestamp"].to_numpy(),
            "actual_label_offline_only": run_scores["failure"].to_numpy(),
            "phase": run_scores["phase"].to_numpy(),
            "model_risk": run_scores["model_risk"].to_numpy(),
            "t2": run_scores["t2"].to_numpy(),
            "q_spe": run_scores["q_spe"].to_numpy(),
            "ewma_alert": run_scores["ewma_alert"].to_numpy(),
            "cusum_alert": run_scores["cusum_alert"].to_numpy(),
            "mspc_alert": run_scores["mspc_alert"].to_numpy(),
            "top_contributing_anonymous_variables": top_variables,
            "review_priority": priority,
            "review_priority_score": priority_score.to_numpy(),
        }
    )
