"""Read-only evidence browser for the current retrospective study."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/study"
st.set_page_config(page_title="SECOM | Risk, Drift & Review", layout="wide")
st.title("Failure risk under process drift")
st.caption("SECOM materials MSc portfolio study · retrospective public-data analysis · AI-assisted implementation")
if not (OUT / "summary.json").exists():
    st.error("Evidence is not generated yet. Run: python scripts/run_study.py")
    st.stop()
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

def table(name):
    return pd.read_csv(OUT / f"{name}.csv")

overview, validation, monitoring, cases, simulation, evidence = st.tabs(
    ["Research overview", "Time validation", "Why alarms fail", "Run cases", "Known-change simulation", "Evidence & limits"])
with overview:
    st.markdown("**Question:** Can a model learned from historical manufacturing measurements remain useful when the process changes?")
    cols = st.columns(4)
    cols[0].metric("Measurements / runs", f"{summary['data']['features']} / {summary['data']['rows']:,}")
    cols[1].metric("Failures", summary["data"]["failures"])
    cols[2].metric("Latest-period AP", f"{summary['latest_metrics']['average_precision']:.3f}")
    cols[3].metric("Latest-period prevalence", f"{summary['latest_metrics']['failure_rate']:.1%}")
    st.image(str(OUT / "figures/study_overview.png"))
    st.info("All periods were previously inspected. These are historical backtests, not new external validation. The grey band is a random-review reference range, not a confidence interval for model performance.")
    st.dataframe(table("illustrative_policy_cost"), hide_index=True)
    st.caption("Illustrative cost: missed failure = 10, passing run reviewed = 1. No factory costs were measured.")
with validation:
    st.markdown("Each outer window selects a model and threshold using earlier expanding-window validation. Preprocessing is refitted inside each training period. Fixed ablations do not choose the policy.")
    st.dataframe(table("splits"), hide_index=True)
    st.dataframe(table("temporal_metrics"), hide_index=True)
    st.subheader("Fixed-model random-split diagnostic")
    st.dataframe(table("random_split_diagnostic"), hide_index=True)
    st.caption("Random and time splits contain different examples; this comparison alone cannot prove the cause of a performance difference.")
    st.subheader("Latest-period conditional uncertainty")
    st.dataframe(table("conditional_day_bootstrap"), hide_index=True)
    st.caption("Whole observed days resampled with replacement, model held fixed. Few days, possible dependence across days, and historical reuse limit inference.")
    st.subheader("Review capacity")
    cap = table("capacity")
    st.dataframe(cap[cap.model.eq("selected_policy")], hide_index=True)
    st.caption("Retrospective batch ranking. Random-null probabilities are descriptive and unadjusted for multiple comparisons. Scores are not calibrated probabilities.")
with monitoring:
    st.image(str(OUT / "figures/alarm_diagnosis.png"))
    st.markdown("The two plotted sensors were chosen after the v2 audit. They are diagnostic examples, not prospectively discovered physical causes. I-chart lines use fitted limits; I-MR also tests moving ranges.")
    st.dataframe(table("sensor_diagnostics"), hide_index=True)
    st.subheader("One variable versus a union of alarms")
    st.dataframe(table("alarm_union_diagnostics"), hide_index=True)
    st.subheader("Reference sensitivity — no winner promoted")
    st.dataframe(table("reference_sensitivity"), hide_index=True)
    st.caption("Sensitivity runs reset monitoring at the evaluation boundary. Main runs carry post-reference state. Label-assisted pass references are not verified in-control production.")
with cases:
    chosen = table("case_studies")
    st.caption("Earliest timestamp in each available outcome category under fixed 20% batch review. A category with no examples is not fabricated.")
    case_id = st.selectbox("Illustrative case", chosen.run_id.tolist(), format_func=lambda rid: f"Run {rid} — {chosen.loc[chosen.run_id.eq(rid), 'case_type'].iloc[0]}")
    row = chosen.loc[chosen.run_id.eq(case_id)].iloc[0]
    st.dataframe(row.astype(str).to_frame("Observed value"))
    st.warning("Q residual top-three variables explain PCA reconstruction error only. They do not explain classifier risk, T², CUSUM or a physical failure mechanism.")
    st.markdown("**What an engineer would need next:** sensor identity and units, recipe/tool/lot context, maintenance history, measurement quality, and an independent physical investigation. None are supplied by SECOM.")
    st.download_button("Download latest-period run evidence", (OUT / "final_run_review.csv").read_bytes(), "final_run_review.csv", "text/csv")
with simulation:
    st.image(str(OUT / "figures/simulation.png"))
    st.dataframe(table("simulation_summary"), hide_index=True)
    st.caption(summary["simulation"])
    st.info("Pre-change false alarms are reported separately; state is not reset after alarms. Detection delay does not establish real manufacturing warning lead time.")
with evidence:
    st.markdown("""
- Public anonymous data; no measured yield improvement, causal diagnosis or production deployment.
- No wafer/lot/tool IDs, sensor units or specification limits; no Cp/Cpk claim.
- Model risk and process excursion scores answer different questions.
- Implementation and documentation were developed with AI assistance; independent author mastery has not been assessed.
- Historical v2 outputs remain in reports/legacy_v2; they are not current validated results.
""")
    st.json({"study_protocol": summary["scope"], "decisions": summary["decisions"]})
    st.download_button("Download provenance manifest", (OUT / "manifest.json").read_bytes(), "manifest.json", "application/json")
