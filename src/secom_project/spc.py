"""Leakage-safe Phase I/II statistical process monitoring utilities.

The Phase I reference may be restricted to historically labelled pass runs.
That is a label-assisted approximation, not evidence that the period was a
validated in-control production regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class PhaseDefinition:
    """Chronological row positions used by the monitoring workflow."""

    train_size: int
    phase_i_size: int
    reference_positions: np.ndarray
    phase_labels: np.ndarray


def define_phases(
    y_ordered: pd.Series,
    train_size: int,
    phase_i_fraction: float = 0.60,
) -> PhaseDefinition:
    """Create Phase I, Phase II and final-holdout positions.

    Only known pass runs inside the earliest Phase I window form the reference.
    Holdout labels are never consulted.
    """

    n_rows = len(y_ordered)
    if not 1 < train_size < n_rows:
        raise ValueError("train_size must leave non-empty training and holdout data.")
    if not 0 < phase_i_fraction < 1:
        raise ValueError("phase_i_fraction must be between zero and one.")
    phase_i_size = max(2, int(np.floor(train_size * phase_i_fraction)))
    phase_i_labels = y_ordered.iloc[:phase_i_size].to_numpy()
    reference_positions = np.flatnonzero(phase_i_labels == 0)
    if len(reference_positions) < 3:
        raise ValueError("Phase I requires at least three label-assisted pass runs.")
    phases = np.full(n_rows, "final_holdout", dtype=object)
    phases[:phase_i_size] = "phase_i"
    phases[phase_i_size:train_size] = "phase_ii"
    return PhaseDefinition(train_size, phase_i_size, reference_positions, phases)


def _standardized_effect(pass_values: pd.Series, fail_values: pd.Series) -> float:
    pass_clean = pass_values.dropna().to_numpy(dtype=float)
    fail_clean = fail_values.dropna().to_numpy(dtype=float)
    if len(pass_clean) < 2 or len(fail_clean) < 2:
        return 0.0
    pooled = np.sqrt((np.var(pass_clean, ddof=1) + np.var(fail_clean, ddof=1)) / 2)
    return float((np.mean(fail_clean) - np.mean(pass_clean)) / pooled) if pooled > 0 else 0.0


def select_monitoring_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 15,
    max_missing: float = 0.50,
) -> list[str]:
    """Select stable monitoring variables using training labels only."""

    if not 10 <= top_n <= 20:
        raise ValueError("top_n must be between 10 and 20.")
    eligible = X_train.columns[X_train.isna().mean() <= max_missing]
    windows = np.array_split(np.arange(len(X_train)), 4)
    rows: list[tuple[str, float, float]] = []
    for feature in eligible:
        effects = []
        for positions in windows:
            labels = y_train.iloc[positions]
            values = X_train.iloc[positions][feature]
            effects.append(
                _standardized_effect(values[labels.to_numpy() == 0], values[labels.to_numpy() == 1])
            )
        absolute = np.abs(effects)
        rows.append((str(feature), float(np.mean(absolute)), float(np.std(absolute))))
    ranking = pd.DataFrame(rows, columns=["feature", "mean_effect", "effect_variability"])
    ranking["stable_score"] = ranking["mean_effect"] / (1 + ranking["effect_variability"])
    return ranking.sort_values(
        ["stable_score", "mean_effect", "feature"], ascending=[False, False, True]
    ).head(top_n)["feature"].tolist()


class UnivariateSPC:
    """Reference-fitted I-MR, EWMA and two-sided CUSUM monitoring."""

    def __init__(self, ewma_lambda: float = 0.20, ewma_l: float = 3.0, cusum_k: float = 0.5, cusum_h: float = 5.0):
        self.ewma_lambda = ewma_lambda
        self.ewma_l = ewma_l
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

    def fit(self, X_reference: pd.DataFrame) -> "UnivariateSPC":
        if X_reference.empty:
            raise ValueError("Reference data cannot be empty.")
        self.features_ = list(X_reference.columns)
        self.medians_ = X_reference.median().fillna(0.0)
        filled = X_reference.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=1).replace(0, 1.0).fillna(1.0)
        reference_z = (filled - self.means_) / self.scales_
        mean_moving_range = reference_z.diff().abs().mean().fillna(0.0)
        self.imr_i_limits_ = (2.66 * mean_moving_range).replace(0, 3.0)
        self.mr_limits_ = mean_moving_range * 3.267
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = X.loc[:, self.features_].fillna(self.medians_)
        z = (values - self.means_) / self.scales_
        output: dict[str, np.ndarray] = {}
        n_rows = len(z)
        time_index = np.arange(1, n_rows + 1)
        ewma_limit = self.ewma_l * np.sqrt(
            self.ewma_lambda / (2 - self.ewma_lambda)
            * (1 - (1 - self.ewma_lambda) ** (2 * time_index))
        )
        for feature in self.features_:
            x = z[feature].to_numpy(dtype=float)
            mr = np.r_[0.0, np.abs(np.diff(x))]
            ewma = np.empty(n_rows, dtype=float)
            positive = np.empty(n_rows, dtype=float)
            negative = np.empty(n_rows, dtype=float)
            previous_ewma = previous_pos = previous_neg = 0.0
            for index, value in enumerate(x):
                previous_ewma = self.ewma_lambda * value + (1 - self.ewma_lambda) * previous_ewma
                previous_pos = max(0.0, previous_pos + value - self.cusum_k)
                previous_neg = min(0.0, previous_neg + value + self.cusum_k)
                ewma[index] = previous_ewma
                positive[index] = previous_pos
                negative[index] = previous_neg
            output[f"{feature}__z"] = x
            output[f"{feature}__imr_alert"] = (np.abs(x) > self.imr_i_limits_[feature]) | (mr > self.mr_limits_[feature])
            output[f"{feature}__ewma"] = ewma
            output[f"{feature}__ewma_alert"] = np.abs(ewma) > ewma_limit
            output[f"{feature}__cusum"] = np.maximum(positive, -negative)
            output[f"{feature}__cusum_alert"] = (positive > self.cusum_h) | (-negative > self.cusum_h)
        frame = pd.DataFrame(output, index=X.index)
        for method in ("imr", "ewma", "cusum"):
            columns = [f"{feature}__{method}_alert" for feature in self.features_]
            frame[f"{method}_alert"] = frame[columns].any(axis=1)
        ewma_columns = [f"{feature}__ewma" for feature in self.features_]
        cusum_columns = [f"{feature}__cusum" for feature in self.features_]
        frame["ewma_score"] = frame[ewma_columns].abs().max(axis=1) / max(float(ewma_limit[-1]), 1e-12)
        frame["cusum_score"] = frame[cusum_columns].max(axis=1) / self.cusum_h
        return frame


class MultivariateSPC:
    """Reference-only StandardScaler/PCA with T2 and Q/SPE limits."""

    def __init__(self, variance_target: float = 0.90, max_components: int = 20, limit_quantile: float = 0.99, random_state: int = RANDOM_STATE):
        self.variance_target = variance_target
        self.max_components = max_components
        self.limit_quantile = limit_quantile
        self.random_state = random_state

    def fit(self, X_reference: pd.DataFrame) -> "MultivariateSPC":
        self.features_ = list(X_reference.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        imputed = self.imputer_.fit_transform(X_reference)
        scaled = self.scaler_.fit_transform(imputed)
        cap = min(self.max_components, scaled.shape[1], scaled.shape[0] - 1)
        pilot = PCA(n_components=cap, svd_solver="full", random_state=self.random_state).fit(scaled)
        cumulative = np.cumsum(pilot.explained_variance_ratio_)
        self.n_components_ = int(min(cap, np.searchsorted(cumulative, self.variance_target) + 1))
        self.pca_ = PCA(n_components=self.n_components_, svd_solver="full", random_state=self.random_state).fit(scaled)
        t2, q, _ = self._scores_from_scaled(scaled)
        self.t2_limit_ = float(np.quantile(t2, self.limit_quantile))
        self.q_limit_ = float(np.quantile(q, self.limit_quantile))
        self.threshold_method_ = f"empirical Phase I reference quantile q={self.limit_quantile:.3f}"
        return self

    def _scores_from_scaled(self, scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = self.pca_.transform(scaled)
        variance = np.maximum(self.pca_.explained_variance_, 1e-12)
        t2 = np.sum((scores**2) / variance, axis=1)
        reconstructed = self.pca_.inverse_transform(scores)
        residual = scaled - reconstructed
        q = np.sum(residual**2, axis=1)
        return np.maximum(t2, 0), np.maximum(q, 0), residual**2

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler_.transform(self.imputer_.transform(X.loc[:, self.features_]))
        t2, q, residual_sq = self._scores_from_scaled(scaled)
        return pd.DataFrame(
            {
                "t2": t2,
                "q_spe": q,
                "t2_alert": t2 > self.t2_limit_,
                "q_alert": q > self.q_limit_,
                "mspc_alert": (t2 > self.t2_limit_) | (q > self.q_limit_),
                "t2_ratio": t2 / max(self.t2_limit_, 1e-12),
                "q_ratio": q / max(self.q_limit_, 1e-12),
            },
            index=X.index,
        ).assign(_residual_contributions=list(residual_sq))

    def q_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler_.transform(self.imputer_.transform(X.loc[:, self.features_]))
        _, _, residual_sq = self._scores_from_scaled(scaled)
        return pd.DataFrame(residual_sq, columns=self.features_, index=X.index)


def alert_metrics(y_true: pd.Series | np.ndarray, alerts: pd.Series | np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    flag = np.asarray(alerts, dtype=bool)
    tp = int(np.sum(flag & (truth == 1)))
    fp = int(np.sum(flag & (truth == 0)))
    failures = int(np.sum(truth == 1))
    passes = int(np.sum(truth == 0))
    alert_count = int(flag.sum())
    precision = tp / alert_count if alert_count else 0.0
    recall = tp / failures if failures else 0.0
    alert_rate = alert_count / len(truth)
    random_capture = alert_rate
    pass_alerts = flag[truth == 0]
    false_alert_positions = np.flatnonzero(pass_alerts)
    if len(false_alert_positions) >= 2:
        pass_between = float(np.mean(np.diff(false_alert_positions) - 1))
    elif len(false_alert_positions) == 1:
        pass_between = float(passes - 1)
    else:
        pass_between = float(passes)
    return {
        "alert_rate": alert_rate,
        "failure_recall": recall,
        "failure_capture": recall,
        "precision": precision,
        "false_positive_rate": fp / passes if passes else 0.0,
        "alerts_per_100_runs": 100 * alert_rate,
        "failure_capture_lift_over_random": recall / random_capture if random_capture else 0.0,
        "pass_runs_between_false_alerts": pass_between,
        "alerts": alert_count,
        "failures_captured": tp,
        "false_alerts": fp,
    }
