import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from secom_project.study import time_boundary, windows, capacity_evidence, reference_percentile, choose_threshold, fit_policy
from secom_project.triage import make_run_level_triage
from secom_project.spc import UnivariateSPC
from secom_project.monitoring_study import monitor_history

ROOT = Path(__file__).resolve().parents[1]


def test_equal_timestamps_never_straddle_windows():
    times = pd.Series(pd.to_datetime(["2020-01-01"]*3+["2020-01-02"]*4+["2020-01-03"]*3))
    start, end = windows(times,[.5],[1])[0]
    assert start == 3 and end == 10
    assert times.iloc[start-1] < times.iloc[start]
    with pytest.raises(ValueError):
        time_boundary(times.iloc[::-1],.5)


def test_random_reference_is_exact_and_handles_no_failures():
    table = capacity_evidence([1,1,0,0],[.9,.8,.2,.1],[.5])
    assert table.iloc[0].random_expected_captured == 1
    assert table.iloc[0].random_p_ge_observed_descriptive == pytest.approx(1/6)
    empty = capacity_evidence([0,0],[.2,.1],[.5])
    assert pd.isna(empty.iloc[0].recall)


def test_frozen_percentiles_and_ewma_scores_are_prefix_invariant():
    ref = np.linspace(0,1,20)
    old = reference_percentile(ref,[.2,.5])
    assert np.array_equal(old,reference_percentile(ref,[.2,.5,100])[:2])
    rng=np.random.default_rng(3)
    X=pd.DataFrame({"x":rng.normal(size=80)})
    monitor=UnivariateSPC().fit(X.iloc[:20])
    short=monitor.transform(X.iloc[20:40])
    full=monitor.transform(X.iloc[20:])
    pd.testing.assert_frame_equal(short,full.iloc[:20])


def test_review_none_is_available_even_for_score_one():
    threshold=choose_threshold([0,0,1],[1.,1.,0.],fn_cost=1,fp_cost=10)
    assert threshold > 1


def test_triage_aligns_ids_and_rejects_duplicate_or_missing_ids():
    scores=pd.DataFrame({"run_id":[10,20],"timestamp":["a","b"],"failure":[0,1],"phase":["p","p"],
                         "model_risk":[.1,.2],"spc_score":[1,2],"t2":[1,2],"q_spe":[1,2],
                         "ewma_alert":[False,True],"cusum_alert":[False,True],"mspc_alert":[False,True]})
    c=pd.DataFrame({"run_id":[10,20],"x":[9,0],"y":[0,9]})
    expected=make_run_level_triage(scores,c)
    pd.testing.assert_frame_equal(expected,make_run_level_triage(scores,c.iloc[::-1]))
    for ids in ([10,10],[10,30]):
        with pytest.raises(ValueError):
            make_run_level_triage(scores,c.assign(run_id=ids))


def test_monitor_fitting_ignores_later_labels():
    config=json.loads((ROOT/"configs/study.json").read_text())
    rng=np.random.default_rng(8)
    features=[f"sensor_{i:03d}" for i in range(20)]+["sensor_542","sensor_543"]
    X=pd.DataFrame(rng.normal(size=(100,22)),columns=features)
    y=pd.Series(([0]*8+[1]*2)*10)
    times=pd.Series(pd.date_range("2020-01-01",periods=100,freq="h"))
    first=monitor_history(X,y,times,80,config)
    flipped=y.copy();flipped.iloc[48:]=1-flipped.iloc[48:]
    second=monitor_history(X,flipped,times,80,config)
    pd.testing.assert_frame_equal(first[0],second[0])
    pd.testing.assert_frame_equal(first[2],second[2])
    assert first[7] == second[7]


def test_policy_inner_models_only_fit_earlier_rows(monkeypatch):
    from sklearn.base import BaseEstimator, ClassifierMixin
    fits=[]
    class Recorder(ClassifierMixin,BaseEstimator):
        def fit(self,X,y):
            fits.append(list(X.index));self.p_=float(np.mean(y));return self
        def predict_proba(self,X):
            assert min(X.index)>max(fits[-1])
            return np.tile([1-self.p_,self.p_],(len(X),1))
    monkeypatch.setattr("secom_project.study.study_models",lambda seed:{"simple":Recorder()})
    config=json.loads((ROOT/"configs/study.json").read_text());config["candidates"]=["simple"]
    X=pd.DataFrame({"x":np.arange(100)});y=pd.Series([0,1]*50)
    times=pd.Series(pd.date_range("2020-01-01",periods=100,freq="h"))
    _,name,threshold,comparison,_=fit_policy(X,y,times,config)
    assert [len(x) for x in fits] == [50,65,80,100]
    assert name == "simple" and len(comparison)==3
