"""Illustrative return-risk scoring for the dashboard demo."""

from __future__ import annotations

import pandas as pd


def build_demo_return_risk_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Return a rule-based 0–100 demo score, not a return probability."""
    required = {"price", "quantity", "payment_method", "category_name"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for demo risk scoring: {sorted(missing)}")

    scored = data.copy()
    price = pd.to_numeric(scored["price"], errors="raise")
    quantity = pd.to_numeric(scored["quantity"], errors="raise")
    if price.isna().any() or quantity.isna().any():
        raise ValueError("Price and quantity must not contain missing values.")
    if scored[["payment_method", "category_name"]].isna().any().any():
        raise ValueError("Payment method and category must not contain missing values.")

    score = pd.Series(10, index=scored.index, dtype="int64")
    score += price.ge(350).astype("int64") * 25
    score += quantity.ge(4).astype("int64") * 20
    score += scored["payment_method"].astype(str).str.casefold().str.contains(
        "cash on delivery"
    ).astype("int64") * 20
    score += scored["category_name"].astype(str).str.casefold().eq(
        "fashion"
    ).astype("int64") * 25

    scored["demo_return_risk_score"] = score.clip(0, 100)
    scored["demo_risk_band"] = scored["demo_return_risk_score"].ge(60).map(
        {True: "High (demo)", False: "Standard (demo)"}
    )
    return scored
