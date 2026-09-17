"""Known-change simulation: validates monitoring behaviour, not fab performance."""
import numpy as np
import pandas as pd
from .spc import UnivariateSPC


def simulate_monitoring(config):
    rng = np.random.default_rng(config["seed"])
    repeats = config["simulation_repeats"]
    pre, post = config["simulation_prechange_rows"], config["simulation_postchange_rows"]
    rows, examples = [], []
    for repeat in range(repeats):
        ref = pd.DataFrame({"signal": rng.normal(size=config["simulation_reference_rows"])})
        fitted = UnivariateSPC().fit(ref)
        noise = rng.normal(size=pre + post)
        for scenario in ("stable", "step_1sigma", "ramp_to_2sigma"):
            offset = np.zeros(pre + post)
            if scenario == "step_1sigma":
                offset[pre:] = 1.0
            elif scenario == "ramp_to_2sigma":
                offset[pre:] = np.linspace(0, 2, post)
            values = noise + offset
            scores = fitted.transform(pd.DataFrame({"signal": values}))
            if repeat == 0:
                for i, value in enumerate(values):
                    examples.append({"scenario": scenario, "step": i, "signal": value,
                                     "injected_mean": offset[i],
                                     "i_limit_z": fitted.imr_i_limits_["signal"]})
            for method in ("imr", "ewma", "cusum"):
                flags = scores[method + "_alert"].to_numpy()
                detected = np.flatnonzero(flags[pre:])
                is_change = scenario != "stable"
                rows.append({"repeat": repeat, "scenario": scenario, "method": method,
                             "prechange_alert_fraction": flags[:pre].mean(),
                             "prechange_any_alert": bool(flags[:pre].any()),
                             "postchange_alert_fraction": flags[pre:].mean(),
                             "detected_in_horizon": bool(len(detected)) if is_change else np.nan,
                             "delay_if_detected": int(detected[0] + 1) if is_change and len(detected) else np.nan,
                             "restricted_delay": int(detected[0] + 1) if is_change and len(detected) else (post + 1 if is_change else np.nan)})
    trials = pd.DataFrame(rows)
    summary = trials.groupby(["scenario", "method"], sort=False).agg(
        trials=("repeat", "size"), prechange_alert_fraction=("prechange_alert_fraction", "mean"),
        prechange_probability_any_alert=("prechange_any_alert", "mean"),
        postchange_alert_fraction=("postchange_alert_fraction", "mean"),
        detection_probability=("detected_in_horizon", "mean"),
        median_delay_among_detected=("delay_if_detected", "median"),
        mean_restricted_delay=("restricted_delay", "mean"),
    ).reset_index()
    return trials, summary, pd.DataFrame(examples)
