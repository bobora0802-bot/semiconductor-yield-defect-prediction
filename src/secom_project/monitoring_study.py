"""Reference-fitted monitoring and explicitly retrospective failure diagnosis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .spc import UnivariateSPC, MultivariateSPC, select_monitoring_features, alert_metrics
from .study import time_boundary, reference_percentile


def monitor_history(X, y, timestamps, train_end, config):
    ref_end = time_boundary(timestamps.iloc[:train_end], config["reference_fraction"])
    # Label-assisted feature selection ends with the reference window itself.
    features = select_monitoring_features(X.iloc[:ref_end], y.iloc[:ref_end],
                                          top_n=config["monitoring_features"])
    reference_ids = np.flatnonzero(y.iloc[:ref_end].to_numpy() == 0)
    reference = X.iloc[reference_ids][features]
    uni = UnivariateSPC().fit(reference)
    multi = MultivariateSPC().fit(reference)
    # Monitor only AFTER reference fitting; no retrospective Phase I scores in state.
    future = X.iloc[ref_end:][features]
    us, ms = uni.transform(future), multi.transform(future).drop(columns="_residual_contributions")
    scores = pd.concat([us, ms], axis=1)
    scores.insert(0, "position", scores.index)
    scores["spc_score"] = scores[["ewma_score", "cusum_score", "t2_ratio", "q_ratio"]].max(axis=1)
    earlier_scores = scores.loc[scores.index < train_end, "spc_score"]
    scores["spc_reference_percentile"] = reference_percentile(earlier_scores, scores.spc_score)
    q = multi.q_contributions(future)
    limits = pd.DataFrame({"feature": features, "mean": uni.means_, "scale": uni.scales_,
                           "i_limit_z": uni.imr_i_limits_, "mr_limit_z": uni.mr_limits_}).reset_index(drop=True)
    final = scores.loc[scores.index >= train_end]
    methods = []
    for method, column in [("I-MR", "imr_alert"), ("EWMA", "ewma_alert"),
                           ("CUSUM", "cusum_alert"), ("T2", "t2_alert"), ("Q/SPE", "q_alert")]:
        methods.append({"method": method, **alert_metrics(y.iloc[train_end:], final[column])})
    # Diagnostics cover all selected variables plus the two v2 audit discoveries.
    # These two are post-hoc case studies; they never enter supervised model selection.
    diagnostic_features = list(dict.fromkeys(features + ["sensor_542", "sensor_543"]))
    diagnostic_ref = X.iloc[reference_ids][diagnostic_features]
    du = UnivariateSPC().fit(diagnostic_ref)
    ds = du.transform(X.iloc[ref_end:][diagnostic_features])
    rows = []
    for f in diagnostic_features:
        ref, later = diagnostic_ref[f], X.iloc[train_end:][f]
        z = (later.fillna(du.medians_[f]) - du.means_[f]) / du.scales_[f]
        labels = y.iloc[train_end:]
        ref_clean, later_clean = ref.dropna(), later.dropna()
        alert = ds.loc[ds.index >= train_end, f + "__imr_alert"].to_numpy()
        rows.append({"feature": f, "in_monitoring_set": f in features,
                     "reference_missing": ref.isna().mean(), "later_missing": later.isna().mean(),
                     "reference_unique_values": ref.nunique(), "reference_std": du.scales_[f],
                     "reference_lag1_correlation": ref.autocorr(),
                     "i_limit_z": du.imr_i_limits_[f],
                     "later_pass_mean_z": z[labels == 0].mean(),
                     "later_failure_mean_z": z[labels == 1].mean(),
                     "ks_statistic": ks_2samp(ref_clean, later_clean).statistic if len(ref_clean) and len(later_clean) else np.nan,
                     **alert_metrics(labels, alert)})
    diagnostics = pd.DataFrame(rows)
    # Same variables, different historical reference policies. Never pick a winner.
    references = {
        "early_pass": reference_ids,
        "early_all": np.arange(ref_end),
        "recent_pass": np.flatnonzero(y.iloc[:train_end].to_numpy() == 0),
    }
    references["recent_pass"] = references["recent_pass"][references["recent_pass"] >= ref_end]
    sensitivity = []
    for name, ids in references.items():
        fitted = UnivariateSPC().fit(X.iloc[ids][features])
        # Reset state at the same evaluation boundary for a fair reference-only contrast.
        sc = fitted.transform(X.iloc[train_end:][features])
        for method in ("imr", "ewma", "cusum"):
            sensitivity.append({"reference_policy": name, "reference_rows": len(ids),
                                "method": method, **alert_metrics(y.iloc[train_end:], sc[method + "_alert"])})
    union_rows = []
    for method in ("imr", "ewma", "cusum"):
        alert_matrix = final[[f + f"__{method}_alert" for f in features]]
        rates = alert_matrix.mean().sort_values(ascending=False)
        dominant = rates.index[0]
        union_rows.append({"method": method, "dominant_variable": dominant.split("__")[0],
                           "dominant_alert_rate": rates.iloc[0],
                           "union_alert_rate": alert_matrix.any(axis=1).mean(),
                           "union_without_dominant": alert_matrix.drop(columns=dominant).any(axis=1).mean()})
    metadata = {"reference_end": ref_end, "reference_rows": len(reference_ids),
                "features": features, "reference_positions": reference_ids.tolist(),
                "pca_components": multi.n_components_, "t2_limit": multi.t2_limit_, "q_limit": multi.q_limit_,
                "state_policy": "Start after reference; carry EWMA/CUSUM state into later period. Sensitivity experiment resets at evaluation start.",
                "reference_policy": "historically labelled passes; not a validated in-control regime"}
    return scores, q, limits, pd.DataFrame(methods), diagnostics, pd.DataFrame(sensitivity), pd.DataFrame(union_rows), metadata, du
