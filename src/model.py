from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.processor import build_preprocessor
from src.settings import MLFLOW_TRACKING_URI


class Model:
    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        self.estimator = estimator
        self.pipeline: Pipeline | None = None
        self.training_metrics: dict[str, float] = {}

    def _build_estimator(self, y: pd.Series) -> BaseEstimator:
        if pd.api.types.is_numeric_dtype(y):
            return LinearRegression()
        return LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if self.estimator is None:
            self.estimator = self._build_estimator(y)
        preprocessor = build_preprocessor(X)
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor.named_steps["preprocessor"]),
                ("model", self.estimator),
            ]
        )
        self.pipeline.fit(X, y)
        preds = self.pipeline.predict(X)
        self.training_metrics = self._evaluate(y, preds)
        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")
        predictions = self.pipeline.predict(X)
        return pd.Series(predictions, index=X.index)

    def save(self, path: str | Path) -> None:
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str | Path) -> None:
        self.pipeline = joblib.load(path)

    def _evaluate(self, y_true: pd.Series, y_pred: pd.Series | list[float]) -> dict[str, float]:
        if pd.api.types.is_numeric_dtype(y_true):
            return {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                "r2": r2_score(y_true, y_pred),
            }

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }


def train_with_tracking(
    X: pd.DataFrame,
    y: pd.Series,
    tracking_uri: str | None = None,
) -> tuple[Model, dict[str, float]]:
    import mlflow
    import mlflow.sklearn

    uri = tracking_uri or MLFLOW_TRACKING_URI
    if uri:
        mlflow.set_tracking_uri(uri)

    model = Model()
    with mlflow.start_run():
        metrics = model.fit(X, y)
        mlflow.log_param("estimator", type(model.estimator).__name__)
        if model.estimator is not None:
            params = model.estimator.get_params(deep=False)
            mlflow.log_params(
                {
                    key: value if isinstance(value, (str, int, float, bool)) else str(value)
                    for key, value in params.items()
                }
            )
        mlflow.log_metrics(metrics)
        if model.pipeline is None:
            raise RuntimeError("Training completed without creating a model pipeline.")
        mlflow.sklearn.log_model(model.pipeline, artifact_path="model")

    return model, metrics
