"""Tests for preparing the retail demo data."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src import retail, settings


def test_build_retail_datasets_keeps_only_rows_with_numeric_review_scores() -> None:
    source = pd.DataFrame(
        {
            "order_date": ["2025-01-03", "2025-04-12", "2025-06-02"],
            "review_score": [2.0, 3.0, None],
            "quantity": [1, 2, 1],
            "price": [12.5, 18.0, 8.0],
            "age": [29, 34, 25],
            "category_name": ["Books", "Fashion", "Books"],
            "payment_method": ["Cash", "Card", "Cash"],
        }
    )

    training, inference = retail.build_retail_datasets(source)

    assert len(training) == 2
    assert training[settings.TARGET_COLUMN].tolist() == [2.0, 3.0]
    assert pd.isna(inference[settings.TARGET_COLUMN].iloc[2])
    assert inference["source_row"].tolist() == [1, 2, 3]
    assert training["price"].dtype == "float64"
    assert training["quantity"].dtype == "int64"
    assert training["age"].dtype == "int64"
    assert inference[settings.RETURN_TARGET_COLUMN].isna().all()


def test_build_retail_datasets_preserves_and_validates_return_labels() -> None:
    source = pd.DataFrame(
        {
            "review_score": [4.0, 2.0, 5.0],
            "quantity": [1, 2, 1],
            "price": [12.5, 18.0, 8.0],
            "age": [29, 34, 25],
            "category_name": ["Books", "Fashion", "Books"],
            "payment_method": ["Cash", "Card", "Cash"],
            "is_returned": [0, 1, None],
        }
    )

    training, inference = retail.build_retail_datasets(source)

    assert training[settings.RETURN_TARGET_COLUMN].iloc[:2].tolist() == [0, 1]
    assert pd.isna(training[settings.RETURN_TARGET_COLUMN].iloc[2])
    assert inference[settings.RETURN_TARGET_COLUMN].iloc[:2].tolist() == [0, 1]
    assert pd.isna(inference[settings.RETURN_TARGET_COLUMN].iloc[2])


def test_build_retail_datasets_rejects_non_binary_return_labels() -> None:
    source = pd.DataFrame(
        {
            "review_score": [4.0],
            "quantity": [1],
            "price": [12.5],
            "age": [29],
            "category_name": ["Books"],
            "payment_method": ["Cash"],
            "is_returned": [2],
        }
    )

    with pytest.raises(ValueError, match="is_returned values must be 0, 1, or blank"):
        retail.build_retail_datasets(source)


def test_build_retail_datasets_preserves_existing_source_row() -> None:
    source = pd.DataFrame(
        {
            "source_row": [101],
            "order_date": ["2025-01-03"],
            "review_score": [5.0],
            "price": [12.5],
            "quantity": [2],
            "category_name": ["Books"],
            "payment_method": ["Cash"],
            "age": [29],
        }
    )

    training, inference = retail.build_retail_datasets(source)

    assert training["source_row"].tolist() == [101]
    assert inference["source_row"].tolist() == [101]


def test_import_retail_data_sends_only_typed_columns(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "retail.csv"
    pd.DataFrame(
        {
            "price": ["12.50"],
            "quantity": ["2"],
            "category_name": ["Books"],
            "payment_method": ["Cash"],
            "age": ["29"],
            "review_score": [None],
            "unselected_column": ["not imported"],
        }
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(settings, "RETAIL_DATA_PATH", str(source_path))

    with patch("src.database.save_retail_sales") as save:
        count = retail.import_retail_data()

    assert count == 1
    rows = save.call_args.args[0]
    assert set(rows[0]) == {
        "source_row",
        "price",
        "quantity",
        "category_name",
        "payment_method",
        "age",
        "review_score",
        "is_returned",
    }
    assert rows[0]["price"] == 12.5
    assert rows[0]["quantity"] == 2
    assert rows[0]["age"] == 29
    assert rows[0]["review_score"] is None
    assert rows[0]["is_returned"] is None
