"""Tests for the illustrative return-risk score."""

from __future__ import annotations

import pandas as pd
import pytest

from src.return_risk import build_demo_return_risk_scores


def test_build_demo_return_risk_scores_applies_documented_rules() -> None:
    data = pd.DataFrame(
        {
            "price": [100, 400],
            "quantity": [1, 4],
            "payment_method": ["Card", "Cash on Delivery"],
            "category_name": ["Books", "Fashion"],
        }
    )

    scored = build_demo_return_risk_scores(data)

    assert scored["demo_return_risk_score"].tolist() == [10, 100]
    assert scored["demo_risk_band"].tolist() == ["Standard (demo)", "High (demo)"]
    assert "demo_return_risk_score" not in data.columns


def test_build_demo_return_risk_scores_rejects_missing_columns() -> None:
    data = pd.DataFrame({"price": [10]})

    with pytest.raises(ValueError, match="Missing columns"):
        build_demo_return_risk_scores(data)


def test_build_demo_return_risk_scores_rejects_missing_numeric_values() -> None:
    data = pd.DataFrame(
        {
            "price": [None],
            "quantity": [1],
            "payment_method": ["Card"],
            "category_name": ["Books"],
        }
    )

    with pytest.raises(ValueError, match="must not contain missing"):
        build_demo_return_risk_scores(data)
