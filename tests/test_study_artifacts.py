"""Cross-artifact consistency checks after the reproduction command."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from secom_project.study import score_metrics, capacity_evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/study"
pytestmark=pytest.mark.skipif(not (OUT/"summary.json").exists(),reason="Generate study artifacts first")


def test_published_metrics_and_cases_recompute_from_predictions():
    pred=pd.read_csv(OUT/"predictions.csv")
    last=pred[pred.window==pred.window.max()]
    summary=json.loads((OUT/"summary.json").read_text())
    m=score_metrics(last.truth,last.risk)
    assert m['average_precision']==pytest.approx(summary['latest_metrics']['average_precision'])
    cap=capacity_evidence(last.truth,last.risk,[r['target_capacity'] for r in summary['latest_capacity']])
    assert cap.captured.tolist()==[r['captured'] for r in summary['latest_capacity']]
    cases=pd.read_csv(OUT/"case_studies.csv")
    all_cases=pd.read_csv(OUT/"final_run_review.csv")
    for row in cases.itertuples():
        assert row.run_id==all_cases[all_cases.case_type==row.case_type].iloc[0].run_id
    assert cases.run_id.is_unique
    q=pd.read_csv(OUT/"q_contributions.csv").set_index('run_id')
    for row in cases.itertuples():
        assert set(row.q_residual_top3_only.split('; '))<=set(q.columns)
        features=summary['monitoring']['features']
        assert q.loc[row.run_id,features].sum()==pytest.approx(row.q_spe)


def test_outputs_have_no_future_fit_boundary_or_duplicate_prediction_ids():
    splits=pd.read_csv(OUT/"splits.csv")
    assert (pd.to_datetime(splits.train_end)<pd.to_datetime(splits.evaluation_start)).all()
    p=pd.read_csv(OUT/"predictions.csv")
    assert p.run_id.is_unique
    assert np.isfinite(p.risk).all()
    assert p.risk.between(0,1).all()


def test_dashboard_runs_and_case_selection_is_functional():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.tabs)==6
    for value in pd.read_csv(OUT/'case_studies.csv').run_id:
        app.selectbox[0].set_value(int(value)).run()
        assert not app.exception
