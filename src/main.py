from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from src.database import save_predictions_to_supabase
from src.extractor import load_inference_data, load_training_data
from src.model import Model, train_with_tracking
from src.output import save_predictions
from src.processor import prepare_training_data
from src.settings import (
    DATA_PATH,
    INFERENCE_DATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    PREDICTIONS_PATH,
)
from src.tuning import tune_random_forest


def _ensure_demo_data() -> None:
    data_dir = Path(DATA_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    if not Path(DATA_PATH).exists():
        df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                "target": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        df.to_csv(DATA_PATH, index=False)

    if not Path(INFERENCE_DATA_PATH).exists():
        inference = pd.DataFrame(
            {
                "feature_1": [9.0, 10.0, 11.0],
                "feature_2": [8.5, 9.5, 10.5],
            }
        )
        inference.to_csv(INFERENCE_DATA_PATH, index=False)


def train() -> None:
    _ensure_demo_data()
    training_df = load_training_data(DATA_PATH)
    X, y = prepare_training_data(training_df)
    if MLFLOW_TRACKING_URI:
        model, metrics = train_with_tracking(X, y)
    else:
        model = Model()
        metrics = model.fit(X, y)
    model.save(MODEL_PATH)
    print(metrics)


def predict() -> None:
    _ensure_demo_data()
    if not Path(MODEL_PATH).exists():
        train()

    inference_df = load_inference_data(INFERENCE_DATA_PATH)
    model = Model()
    model.load(MODEL_PATH)
    predictions = model.predict(inference_df)
    save_predictions(predictions, PREDICTIONS_PATH)
    print(f"Saved {len(predictions)} predictions to {PREDICTIONS_PATH}")

    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        payload = json.loads(PREDICTIONS_PATH.read_text())
        inserted = save_predictions_to_supabase(payload)
        print(f"Inserted {inserted} rows into Supabase table 'predictions'")


def tune() -> None:
    _ensure_demo_data()
    training_df = load_training_data(DATA_PATH)
    X, y = prepare_training_data(training_df)
    study = tune_random_forest(X, y)
    print({"best_params": study.best_params, "best_value": study.best_value})


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [train|predict|tune]")
        return

    command = sys.argv[1]
    if command == "train":
        train()
    elif command == "predict":
        predict()
    elif command == "tune":
        tune()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
