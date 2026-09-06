"""Read-only Streamlit dashboard for generated SECOM evidence artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"


@st.cache_data
def load_artifacts():
    return {
        "baseline": json.loads((REPORTS / "baseline_summary.json").read_text(encoding="utf-8")),
        "enhanced": json.loads((REPORTS / "enhanced_summary.json").read_text(encoding="utf-8")),
        "spc": json.loads((REPORTS / "spc_triage_summary.json").read_text(encoding="utf-8")),
        "runs": pd.read_csv(TABLES / "mspc_run_scores.csv", parse_dates=["timestamp"]),
        "triage": pd.read_csv(TABLES / "run_level_triage.csv", parse_dates=["timestamp"]),
        "candidates": pd.read_csv(TABLES / "candidate_process_variables.csv"),
        "strategies": pd.read_csv(TABLES / "monitoring_strategy_comparison.csv"),
    }


st.set_page_config(page_title="SECOM Excursion Monitoring & Triage", layout="wide")
st.title("Semiconductor Yield Excursion Monitoring & Root-Cause Triage")
st.caption("Offline research dashboard. Reads generated CSV/JSON only; it does not retrain models.")

try:
    data = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Missing generated artifact: {error}. Run the analysis scripts first.")
    st.stop()

overview, monitoring, run_triage, candidate_variables, limitations = st.tabs(
    ["Overview", "Process Monitoring", "Run Triage", "Candidate Variables", "Limitations"]
)

with overview:
    enhanced = data["enhanced"]
    columns = st.columns(4)
    columns[0].metric("Runs", "1,567")
    columns[1].metric("Anonymous variables", "590")
    columns[2].metric("Failures", "104 (6.64%)")
    columns[3].metric("Final chronological holdout", data["spc"]["split"]["final_holdout_rows"])
    comparison = pd.DataFrame(
        [
            {"evaluation": "Random holdout", **enhanced["random_stratified"]["metrics"]},
            {"evaluation": "Chronological holdout", **enhanced["chronological"]["metrics"]},
        ]
    )
    st.subheader("Existing failure-risk model")
    st.dataframe(comparison[["evaluation", "roc_auc", "pr_auc", "precision", "recall"]], hide_index=True, width="stretch")
    st.subheader("Review-capacity strategy comparison")
    st.dataframe(data["strategies"], hide_index=True, width="stretch")
    st.info("Random splitting materially overstates later-period performance. SPC alerts and supervised risk scores answer different operational questions.")

with monitoring:
    runs = data["runs"]
    figure_columns = st.columns(2)
    figure_columns[0].image(str(REPORTS / "figures" / "mspc_t2_timeline.png"), caption="Reference-fitted Hotelling T² timeline")
    figure_columns[1].image(str(REPORTS / "figures" / "mspc_q_timeline.png"), caption="Reference-fitted Q/SPE timeline")
    score_name = st.selectbox("Monitoring statistic", ["t2", "q_spe", "ewma_score", "cusum_score"])
    chart = runs.set_index("timestamp")[[score_name]].copy()
    chart["observed_failure_offline"] = runs.set_index("timestamp")["failure"] * chart[score_name].max()
    st.line_chart(chart)
    st.dataframe(
        runs[["run_id", "timestamp", "phase", "failure", "t2", "q_spe", "ewma_alert", "cusum_alert", "mspc_alert"]].tail(100),
        hide_index=True,
        width="stretch",
    )
    st.caption("Failure markers are available only for offline evaluation; control limits were frozen from the Phase I reference.")

with run_triage:
    triage = data["triage"]
    run_id = st.selectbox("Select run", triage["run_id"].tolist(), index=len(triage) - 1)
    selected = triage.loc[triage["run_id"] == run_id].iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Model risk", f"{selected['model_risk']:.3f}")
    metrics[1].metric("T²", f"{selected['t2']:.2f}")
    metrics[2].metric("Q/SPE", f"{selected['q_spe']:.2f}")
    metrics[3].metric("Review priority", selected["review_priority"])
    st.write("Top contributing anonymous variables:", selected["top_contributing_anonymous_variables"])
    st.json(
        {
            "timestamp": str(selected["timestamp"]),
            "actual_label_offline_only": int(selected["actual_label_offline_only"]),
            "ewma_alert": bool(selected["ewma_alert"]),
            "cusum_alert": bool(selected["cusum_alert"]),
            "mspc_alert": bool(selected["mspc_alert"]),
        }
    )

with candidate_variables:
    candidates = data["candidates"]
    figure_columns = st.columns(2)
    figure_columns[0].image(str(REPORTS / "figures" / "top_sensor_control_charts.png"), caption="Top candidate control-chart signals")
    figure_columns[1].image(str(REPORTS / "figures" / "failure_vs_pass_effect_sizes.png"), caption="Training-only signed effects")
    show = st.slider("Number of candidate process variables", 10, 50, 20)
    st.bar_chart(candidates.head(show).set_index("feature")["consensus_score"])
    st.dataframe(
        candidates.head(show)[
            ["consensus_rank", "feature", "selection_frequency", "top20_frequency", "permutation_importance", "effect_size", "effect_direction", "missing_rate_change", "ks_statistic", "spc_alert_rate", "rank_stability", "interpretation"]
        ],
        hide_index=True,
        width="stretch",
    )

with limitations:
    st.markdown(
        """
- Variables are anonymous, so no pressure, temperature, flow, recipe, or physical mechanism can be identified.
- Lot, wafer, equipment, chamber, recipe, maintenance, and specification-limit identifiers are unavailable.
- Cp/Cpk is intentionally not calculated because real LSL/USL values are unavailable.
- The Phase I pass-only baseline is label-assisted historical reference data, not a validated in-control production period.
- Signals demonstrate association and investigation priority only; they do not establish causality.
- This is an offline research workflow, not an operational monitoring system.
- No claim of physical lead time or genuine early warning is supported by this dataset.
        """
    )
