"""Leakage-safe preprocessing and candidate model definitions."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SparseColumnFilter(BaseEstimator, TransformerMixin):
    """Drop highly incomplete columns using training-fold statistics only."""

    def __init__(self, max_missing: float = 0.50):
        self.max_missing = max_missing

    def fit(self, X: pd.DataFrame, y: object = None) -> "SparseColumnFilter":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SparseColumnFilter expects a pandas DataFrame.")
        if not 0 <= self.max_missing < 1:
            raise ValueError("max_missing must be in [0, 1).")
        missing_rate = X.isna().mean()
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        self.selected_columns_ = missing_rate[missing_rate <= self.max_missing].index.tolist()
        if not self.selected_columns_:
            raise ValueError("The missingness filter removed every feature.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features: object = None):
        return pd.Index(self.selected_columns_).to_numpy(dtype=object)


def _shared_steps(add_indicator: bool = False) -> list[tuple[str, object]]:
    return [
        ("drop_sparse", SparseColumnFilter(max_missing=0.50)),
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("drop_constant", VarianceThreshold()),
    ]


def build_model_pipelines(random_state: int = 42) -> Mapping[str, Pipeline]:
    """Return interpretable linear, PCA, and nonlinear candidate pipelines."""

    logistic_selected = Pipeline(
        _shared_steps()
        + [
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=40)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4_000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logistic_pca = Pipeline(
        _shared_steps()
        + [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=40,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4_000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        _shared_steps()
        + [
            ("select", SelectKBest(score_func=f_classif, k=80)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest_missing_indicator = Pipeline(
        _shared_steps(add_indicator=True)
        + [
            ("select", SelectKBest(score_func=f_classif, k=80)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    histogram_gradient_boosting = Pipeline(
        _shared_steps()
        + [
            ("select", SelectKBest(score_func=f_classif, k=80)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return {
        "logistic_selected": logistic_selected,
        "logistic_pca": logistic_pca,
        "random_forest": random_forest,
        "random_forest_missing_indicator": random_forest_missing_indicator,
        "hist_gradient_boosting": histogram_gradient_boosting,
    }
