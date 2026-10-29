from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def format_predictions(predictions: pd.Series) -> list[dict[str, object]]:
    return [{"prediction": value} for value in predictions.tolist()]


def save_predictions(predictions: pd.Series, path: str | Path) -> None:
    payload = format_predictions(predictions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
