"""Rebuild the current retrospective study from raw data with one command."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secom_project.data import load_secom
from secom_project.study import (windows, fit_policy, study_models, score_metrics,
                                capacity_evidence, batch_selection, reference_percentile)
from secom_project.monitoring_study import monitor_history
from secom_project.simulation import simulate_monitoring


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".py", ".md", ".json", ".csv", ".txt", ".yml", ".ipynb"}:
        data = data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def daily_bootstrap(frame, repeats=500, seed=42):
    """Resample whole observed days; conditional on fitted model, not a refit CI."""
    dates = pd.to_datetime(frame.timestamp).dt.normalize()
    groups = [np.flatnonzero(dates == day) for day in dates.unique()]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = frame.iloc[indices]
        if sample.truth.nunique() != 2:
            continue
        metrics = score_metrics(sample.truth, sample.risk)
        cap = capacity_evidence(sample.truth, sample.risk, [.1, .2], seed)
        rows.append({"average_precision": metrics["average_precision"], "roc_auc": metrics["roc_auc"],
                     "capture_at_10pct": cap.iloc[0].recall, "capture_at_20pct": cap.iloc[1].recall})
    samples = pd.DataFrame(rows)
    return pd.DataFrame([{"metric": column, "lower95": samples[column].quantile(.025),
                          "upper95": samples[column].quantile(.975), "valid_resamples": len(samples),
                          "requested_resamples": repeats, "observed_days": len(groups)} for column in samples])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/study")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    raw = ROOT / "data/raw"
    subprocess.run([sys.executable, str(ROOT / "scripts/download_data.py")], check=True)
    X, y, timestamps = load_secom(raw)
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    original_ids = np.arange(len(X))[order]
    X, y, timestamps = (obj.iloc[order].reset_index(drop=True) for obj in (X, y, timestamps))
    models = study_models(config["seed"])
    outer = windows(timestamps, config["outer_train_fractions"], config["outer_end_fractions"])
    metrics, capacities, predictions, inner_rows, split_rows, decisions = [], [], [], [], [], []
    for fold, (start, end) in enumerate(outer, 1):
        print(f"Outer window {fold}/{len(outer)}: fit before {timestamps.iloc[start]}, evaluate {end-start} rows", flush=True)
        fitted, name, threshold, inner, earlier = fit_policy(X.iloc[:start], y.iloc[:start], timestamps.iloc[:start], config)
        inner.insert(0, "outer_window", fold)
        inner_rows.append(inner)
        decisions.append({"window": fold, "selected_model": name, "threshold": threshold,
                          "fit_end_exclusive": start, "evaluation_end_exclusive": end})
        split_rows.append({"window": fold, "train_rows": start, "evaluation_rows": end-start,
                           "train_start": timestamps.iloc[0], "train_end": timestamps.iloc[start-1],
                           "evaluation_start": timestamps.iloc[start], "evaluation_end": timestamps.iloc[end-1],
                           "train_failures": int(y.iloc[:start].sum()), "evaluation_failures": int(y.iloc[start:end].sum())})
        probability = fitted.predict_proba(X.iloc[start:end])[:, 1]
        evaluated = {"selected_policy": probability, "constant_prevalence": np.full(end-start, y.iloc[:start].mean())}
        # Fixed model comparisons/ablations do NOT feed back into model selection.
        for candidate in config["candidates"] + config["ablations"]:
            model = fitted if candidate == name else clone(models[candidate]).fit(X.iloc[:start], y.iloc[:start])
            evaluated[candidate] = model.predict_proba(X.iloc[start:end])[:, 1]
        for label, prob in evaluated.items():
            metrics.append({"window": fold, "model": label, **score_metrics(y.iloc[start:end], prob)})
            table = capacity_evidence(y.iloc[start:end], prob, config["review_capacities"], config["seed"])
            table.insert(0, "model", label)
            table.insert(0, "window", fold)
            capacities.append(table)
        frozen_pct = reference_percentile(earlier.risk, probability)
        selected_batch = batch_selection(probability, .2, config["seed"])
        predictions.append(pd.DataFrame({"window": fold, "position": np.arange(start,end),
                                         "run_id": original_ids[start:end], "timestamp": timestamps.iloc[start:end].to_numpy(),
                                         "truth": y.iloc[start:end].to_numpy(), "risk": probability,
                                         "threshold": threshold, "threshold_alert": probability >= threshold,
                                         "past_validation_percentile": frozen_pct,
                                         "batch_review_top20": selected_batch}))
        if fold == len(outer):
            final_start, final_predictions = start, predictions[-1]
            # Explanation is an earlier-validation diagnostic, not final-test selection.
            vstart = int(earlier.position.min())
            explain_model = clone(models[name]).fit(X.iloc[:vstart], y.iloc[:vstart])
            perm = permutation_importance(explain_model, X.iloc[vstart:start], y.iloc[vstart:start],
                                          scoring="average_precision", n_repeats=3, random_state=config["seed"], n_jobs=1)
            importance_rows = pd.DataFrame({"feature": X.columns, "mean_ap_decrease": perm.importances_mean,
                                            "std_ap_decrease": perm.importances_std}).sort_values("mean_ap_decrease", ascending=False)
            importance_rows.to_csv(out / "earlier_validation_importance.csv", index=False)
    metric_table, capacity_table = pd.DataFrame(metrics), pd.concat(capacities, ignore_index=True)
    pred_table = pd.concat(predictions, ignore_index=True)
    metric_table.to_csv(out / "temporal_metrics.csv", index=False)
    capacity_table.to_csv(out / "capacity.csv", index=False)
    pred_table.to_csv(out / "predictions.csv", index=False)
    pd.concat(inner_rows, ignore_index=True).to_csv(out / "inner_validation.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / "splits.csv", index=False)
    # A separate fixed-model random split illustrates evaluation design only.
    # No model winner or threshold is imported from this experiment.
    ri, rj = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=config["seed"])
    random_rows = []
    for name in config["candidates"]:
        fitted = clone(models[name]).fit(X.iloc[ri], y.iloc[ri])
        random_rows.append({"model": name, **score_metrics(y.iloc[rj], fitted.predict_proba(X.iloc[rj])[:,1])})
    pd.DataFrame(random_rows).to_csv(out / "random_split_diagnostic.csv", index=False)
    daily_bootstrap(final_predictions, seed=config["seed"]).to_csv(out / "conditional_day_bootstrap.csv", index=False)
    print("Diagnosing reference-fitted monitoring", flush=True)
    scores, q, limits, method_table, diagnostics, sensitivity, unions, monitoring, diagnostic_model = monitor_history(X, y, timestamps, final_start, config)
    for table, filename in [(scores,"monitor_scores"),(limits,"control_limits"),(method_table,"monitor_methods"),
                             (diagnostics,"sensor_diagnostics"),(sensitivity,"reference_sensitivity"),(unions,"alarm_union_diagnostics")]:
        table.to_csv(out / f"{filename}.csv", index=False)
    q.insert(0, "position", q.index)
    q.insert(1, "run_id", original_ids[q.index])
    q.to_csv(out / "q_contributions.csv", index=False)
    # Only final-period cases, all out-of-training. Align by original run ID.
    contributions = q.set_index("run_id").loc[final_predictions.run_id]
    fcols = monitoring["features"]
    top = np.argsort(-contributions[fcols].to_numpy(), axis=1, kind="stable")[:,:3]
    final_cases = final_predictions.copy()
    final_cases["q_residual_top3_only"] = ["; ".join(fcols[i] for i in row) for row in top]
    final_cases["q_spe"] = scores.loc[final_cases.position,"q_spe"].to_numpy()
    final_cases["t2"] = scores.loc[final_cases.position,"t2"].to_numpy()
    final_cases["case_type"] = np.select([
        final_cases.truth.eq(1) & final_cases.batch_review_top20,
        final_cases.truth.eq(1) & ~final_cases.batch_review_top20,
        final_cases.truth.eq(0) & final_cases.batch_review_top20],
        ["captured_failure","missed_failure","false_alert"], default="unreviewed_pass")
    final_cases.to_csv(out / "final_run_review.csv", index=False)
    cases = final_cases[final_cases.case_type != "unreviewed_pass"].groupby("case_type",sort=True).head(1)
    cases.to_csv(out / "case_studies.csv", index=False)
    cost_rows = []
    truth = final_predictions.truth.to_numpy()
    for label, flag in {"frozen_10_to_1_threshold": final_predictions.threshold_alert.to_numpy(),
                        "batch_top20": final_predictions.batch_review_top20.to_numpy(),
                        "review_none": np.zeros(len(truth),bool), "review_all": np.ones(len(truth),bool)}.items():
        fn, fp = int(((truth == 1)&~flag).sum()), int(((truth == 0)&flag).sum())
        cost_rows.append({"policy":label,"missed_failures":fn,"false_alerts":fp,
                          "illustrative_cost":config["false_negative_cost"]*fn+config["false_positive_cost"]*fp})
    pd.DataFrame(cost_rows).to_csv(out / "illustrative_policy_cost.csv", index=False)
    print("Running separate known-change simulations", flush=True)
    trials, sim_summary, sim_examples = simulate_monitoring(config)
    trials.to_csv(out / "simulation_trials.csv", index=False)
    sim_summary.to_csv(out / "simulation_summary.csv", index=False)
    sim_examples.to_csv(out / "simulation_examples.csv", index=False)
    profile = {"rows":len(X),"features":X.shape[1],"failures":int(y.sum()),
               "missing_cells":int(X.isna().sum().sum()),"duplicate_timestamps":int(timestamps.duplicated().sum()),
               "constant_observed_features":int((X.nunique(dropna=True)<=1).sum()),
               "median_gap_seconds":float(timestamps.diff().dt.total_seconds().median()),
               "warning":"No lot/tool IDs; duplicate timestamps are not assumed independent lots."}
    pd.DataFrame({"feature":X.columns,"missing_rate":X.isna().mean(),"observed_unique_values":X.nunique()}).to_csv(out / "data_quality.csv",index=False)
    weekly = pd.DataFrame({"timestamp":timestamps,"failure":y}).set_index("timestamp").resample("7D").failure.agg(["size","sum","mean"])
    weekly.to_csv(out / "weekly_failure_rate.csv")
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for label in ["logistic_selected","random_forest","selected_policy"]:
        sub=metric_table[metric_table.model.eq(label)]
        axes[0].plot(sub.window,sub.average_precision,"o-",label=label)
    base=metric_table[metric_table.model.eq("constant_prevalence")]
    axes[0].plot(base.window,base.failure_rate,"k--",label="failure prevalence")
    axes[0].set(xlabel="Chronological window",ylabel="Average precision",title="Does ranking survive later periods?")
    axes[0].set_xticks(range(1, len(outer)+1))
    axes[0].legend(fontsize=8)
    selected=capacity_table[capacity_table.model.eq("selected_policy") & capacity_table.window.eq(len(outer))]
    axes[1].fill_between(selected.actual_capacity,selected.random_capture_lower95,selected.random_capture_upper95,alpha=.2,color="gray",label="95% random-review range")
    axes[1].plot(selected.actual_capacity,selected.captured,"o-",label="selected policy")
    axes[1].plot(selected.actual_capacity,selected.random_expected_captured,"k--",label="random expectation")
    axes[1].set(xlabel="Fraction reviewed (retrospective batch)",ylabel="Failures captured",title="Latest-period review trade-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures / "study_overview.png",dpi=160);plt.close(fig)
    fig, axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for ax,feature in zip(axes,["sensor_542","sensor_543"],strict=True):
        z=(X[feature].fillna(diagnostic_model.medians_[feature])-diagnostic_model.means_[feature])/diagnostic_model.scales_[feature]
        ax.plot(timestamps,z,lw=.8,label="standardized measurement")
        limit=diagnostic_model.imr_i_limits_[feature]
        ax.axhline(limit,color="firebrick",ls="--",label=f"actual I limit +/-{limit:.3f}")
        ax.axhline(-limit,color="firebrick",ls="--")
        ax.axvline(timestamps.iloc[monitoring["reference_end"]],color="gray",ls=":",label="reference ends")
        ax.axvline(timestamps.iloc[final_start],color="black",ls=":",label="latest period")
        ax.set(ylabel=f"{feature} (z units)");ax.legend(fontsize=8,loc="upper left")
    axes[0].set_title("Post-hoc diagnosis: a shift does not identify a failure mechanism")
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(figures / "alarm_diagnosis.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,(scenario,group) in zip(axes,sim_examples.groupby("scenario",sort=False),strict=True):
        ax.plot(group.step,group.signal,lw=.6,label="synthetic measurement")
        ax.plot(group.step,group.injected_mean,lw=2,label="known mean")
        ax.axvline(config["simulation_prechange_rows"],color="black",ls=":")
        ax.set(title=scenario,xlabel="Observation index")
    axes[0].set_ylabel("Synthetic units");axes[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures / "simulation.png",dpi=160);plt.close(fig)
    summary={"scope":config["interpretation"],"data":profile,"decisions":decisions,"monitoring":monitoring,
             "latest_metrics":metric_table[metric_table.window.eq(len(outer)) & metric_table.model.eq("selected_policy")].to_dict("records")[0],
             "latest_capacity":selected.to_dict("records"),
             "case_selection":"Earliest timestamp in each outcome category at fixed 20% batch review; illustrative, not representative proof.",
             "simulation":"Independent Gaussian noise; unreset monitoring state; delay conditional on known change. Undetected trials censored at horizon+1. No physical fab claims."}
    (out / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    source_paths=[]
    for folder in ("src","scripts","configs","tests","app"):
        source_paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in {".py",".json"})
    packages=["numpy","pandas","scikit-learn","scipy","matplotlib","seaborn","streamlit","nbformat","nbclient","pytest"]
    manifest={"generated_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
              "hash_policy":"Text suffixes .py/.md/.json/.csv/.txt/.yml/.ipynb: UTF-8 without BOM, CRLF normalized to LF. Raw data and binary figures: exact bytes.",
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "git_status":subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True),
              "config":config,"config_sha256":digest(args.config),
              "data_sha256":{p.name:digest(p) for p in raw.glob("secom*") if p.is_file()},
              "source_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(source_paths)},
              "package_versions":{p:importlib.metadata.version(p) for p in packages},
              "artifact_sha256":{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="manifest.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(summary["latest_metrics"],indent=2),flush=True)
    print(f"Study complete: {out}",flush=True)


if __name__ == "__main__":
    main()
