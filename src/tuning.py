from __future__ import annotations

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.processor import build_preprocessor


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> optuna.study.Study:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("RandomForestRegressor tuning requires a numeric target.")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    def objective(trial: optuna.Trial) -> float:
        estimator = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train).named_steps["preprocessor"]),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        return float(mean_absolute_error(y_validation, predictions))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study
