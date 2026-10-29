from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.settings import FEATURE_COLUMNS, TARGET_COLUMN


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    if FEATURE_COLUMNS:
        return list(FEATURE_COLUMNS)
    return [column for column in df.columns if column != TARGET_COLUMN]


def _add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = _select_feature_columns(df)
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN]

    for column in X.columns:
        if X[column].dtype == "O":
            X[column] = X[column].astype(str)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> Pipeline:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns))

    if not transformers:
        raise ValueError("No valid feature columns found for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor)])
