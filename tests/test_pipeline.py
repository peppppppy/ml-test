from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import database, main
from src.model import Model, train_with_tracking
from src.output import format_predictions, save_predictions
from src.processor import prepare_training_data
from src.tuning import tune_random_forest


def _regression_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "category": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], name="target")
    return X, y


def test_model_trains_saves_and_loads_mixed_features(tmp_path: Path) -> None:
    X, y = _regression_data()
    model = Model()

    metrics = model.fit(X, y)
    model_path = tmp_path / "nested" / "model.joblib"
    model.save(model_path)

    loaded_model = Model()
    loaded_model.load(model_path)
    predictions = loaded_model.predict(X)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert len(predictions) == len(y)
    assert predictions.tolist() == pytest.approx(y.tolist())


def test_model_uses_classifier_for_categorical_target() -> None:
    X = pd.DataFrame({"value": [0, 1, 2, 3, 4, 5]})
    y = pd.Series(["low", "low", "low", "high", "high", "high"])
    model = Model()

    metrics = model.fit(X, y)

    assert isinstance(model.estimator, LogisticRegression)
    assert set(metrics) == {"accuracy", "f1"}
    assert model.predict(X).tolist() == y.tolist()


def test_prepare_training_data_uses_configured_target() -> None:
    frame = pd.DataFrame({"feature": [1, 2], "target": [3, 4]})

    X, y = prepare_training_data(frame)

    assert list(X.columns) == ["feature"]
    assert y.tolist() == [3, 4]
    assert "target" in frame.columns


def test_predictions_are_written_as_json(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    predictions = pd.Series(["cat", "dog"])

    save_predictions(predictions, path)

    assert json.loads(path.read_text()) == [
        {"prediction": "cat"},
        {"prediction": "dog"},
    ]
    assert format_predictions(predictions) == [
        {"prediction": "cat"},
        {"prediction": "dog"},
    ]


def test_supabase_inserts_prediction_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTable:
        def __init__(self) -> None:
            self.inserted: list[dict[str, object]] = []

        def insert(self, rows: list[dict[str, object]]) -> FakeTable:
            self.inserted = rows
            return self

        def execute(self) -> Any:
            return type("Response", (), {"data": self.inserted})()

    table = FakeTable()
    monkeypatch.setattr(database, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(database, "SUPABASE_KEY", "test-key")
    monkeypatch.setattr(
        database,
        "create_client",
        lambda *_args: type("Client", (), {"table": lambda _self, _name: table})(),
    )

    inserted = database.save_predictions_to_supabase([{"prediction": 1.5}, {"prediction": 2.5}])

    assert inserted == 2
    assert [row["prediction"] for row in table.inserted] == [1.5, 2.5]
    assert all("T" in str(row["predicted_at"]) for row in table.inserted)


def test_mlflow_tracking_logs_training_run(tmp_path: Path) -> None:
    X, y = _regression_data()
    tracking_uri = (tmp_path / "mlruns").as_uri()

    model, metrics = train_with_tracking(X, y, tracking_uri=tracking_uri)

    assert model.pipeline is not None
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-8)
    assert list((tmp_path / "mlruns").glob("*/meta.yaml"))


def test_optuna_tuning_runs_and_returns_best_parameters() -> None:
    X, y = _regression_data()

    study = tune_random_forest(X, y, n_trials=2)

    assert study.direction.name == "MINIMIZE"
    assert set(study.best_params) == {"n_estimators", "max_depth"}


def test_cli_train_predict_flow_uses_configured_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(main, "DATA_PATH", data_dir / "train.csv")
    monkeypatch.setattr(main, "INFERENCE_DATA_PATH", data_dir / "inference.csv")
    monkeypatch.setattr(main, "MODEL_PATH", data_dir / "model.joblib")
    monkeypatch.setattr(main, "PREDICTIONS_PATH", data_dir / "predictions.json")
    monkeypatch.setattr(main, "MLFLOW_TRACKING_URI", "")

    main.train()
    main.predict()

    assert (data_dir / "model.joblib").exists()
    assert len(json.loads((data_dir / "predictions.json").read_text())) == 3
