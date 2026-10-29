from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_training_data(path: str | Path) -> pd.DataFrame:
    return _load_csv(path)


def load_inference_data(path: str | Path) -> pd.DataFrame:
    return _load_csv(path)
