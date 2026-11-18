"""Train and evaluate lead-time prediction models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE
from data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, feature_matrix

# Lower number = easier to interpret; used as the tie-breaker after accuracy.
INTERPRETABILITY_RANK = {"Linear (Ridge)": 0, "Gradient Boosting": 1, "Random Forest": 2}


@dataclass
class ModelResult:
    best_name: str
    best_pipeline: Pipeline
    metrics: pd.DataFrame
    residual_std: float


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def _candidates() -> dict:
    return {
        "Linear (Ridge)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
    }


def train_models(df: pd.DataFrame) -> ModelResult:
    X, y = feature_matrix(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, estimator in _candidates().items():
        pipe = Pipeline([("prep", _preprocessor()), ("model", estimator)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        })
        fitted[name] = (pipe, y_test - pred)

    metrics = pd.DataFrame(rows)
    # Pick the lowest RMSE; when models are within 1% RMSE of the best, prefer the more interpretable one.
    best_rmse = metrics["RMSE"].min()
    close = metrics[metrics["RMSE"] <= best_rmse * 1.01].copy()
    close["interp"] = close["Model"].map(INTERPRETABILITY_RANK)
    best_name = close.sort_values(["interp", "RMSE"]).iloc[0]["Model"]
    metrics["Selected"] = metrics["Model"] == best_name

    best_pipe, residuals = fitted[best_name]
    # Refit the chosen model on all data for the simulator.
    best_pipe.fit(X, y)
    return ModelResult(best_name, best_pipe, metrics, float(np.std(residuals)))
