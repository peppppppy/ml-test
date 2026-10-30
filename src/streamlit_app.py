"""Streamlit dashboard.

Displays recent predictions stored by the pipeline. Deploy to Streamlit
Community Cloud for a free public URL.

Run locally:  make dashboard
Opens at:     http://localhost:8501
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import database, settings

load_dotenv()

# On Streamlit Community Cloud, credentials live in st.secrets. Copy them into
# env vars so database.py can reach them via os.getenv, same as the pipeline.
try:
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass

st.set_page_config(page_title=settings.DASHBOARD_TITLE, layout="wide")


def main() -> None:
    st.title(settings.DASHBOARD_TITLE)
    st.caption(settings.DASHBOARD_SUBTITLE)

    if not database.is_supabase_configured():
        st.info("Showing the local CSV and prediction files. Configure Supabase to use hosted data.")

    retail_tab, score_tab = st.tabs(["Retail data", "Review score predictions"])
    with retail_tab:
        _render_retail_data()
    with score_tab:
        _render_review_score_predictions()


def _render_retail_data() -> None:
    rows = database.load_retail_sales(limit=5000)
    if not rows:
        st.info("No retail data found. Add data/retail_sales.csv or import rows into Supabase.")
        return

    df = pd.DataFrame(rows)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    st.subheader("Retail data with the selected column types")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders", f"{len(df):,}")
    col2.metric("Units", f"{df['quantity'].sum():,.0f}")
    col3.metric("Average price", f"{df['price'].mean():,.2f}")
    col4.metric("Rated orders", f"{df['review_score'].notna().sum():,}")

    left, right = st.columns(2)
    with left:
        by_category = df.groupby("category_name", as_index=False)["price"].mean()
        st.plotly_chart(
            px.bar(by_category, x="category_name", y="price", title="Average price by category"),
            width="stretch",
        )
    with right:
        by_review = df["review_score"].value_counts().rename_axis("review_score").reset_index(name="orders")
        st.plotly_chart(
            px.bar(by_review, x="review_score", y="orders", title="Review score counts"),
            width="stretch",
        )

    st.subheader("Selected typed columns")
    st.dataframe(
        df.loc[:, ["price", "quantity", "category_name", "payment_method", "age", "review_score"]],
        width="stretch",
    )


def _render_review_score_predictions() -> None:
    rows = database.load_recent_predictions(limit=5000)
    if not rows:
        _render_empty_state()
        return

    df = pd.DataFrame(rows)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    predictions = df["prediction"].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted orders", f"{len(predictions):,}")
    col2.metric("Average predicted review", f"{predictions.mean():.2f} / 5")
    col3.metric("Predicted range", f"{predictions.min():.2f}–{predictions.max():.2f}")

    if "category_name" in df.columns:
        by_category = df.groupby("category_name", as_index=False)["prediction"].mean()
        st.plotly_chart(
            px.bar(
                by_category,
                x="category_name",
                y="prediction",
                title="Average predicted review score by category",
                labels={"prediction": "Review score"},
            ),
            width="stretch",
        )

    st.caption("Predictions are model estimates on a 1–5 scale, not confirmed customer reviews.")
    st.subheader("Review score predictions")
    st.dataframe(df, width="stretch")


def _render_empty_state() -> None:
    st.info(
        "No review-score predictions found. Prepare data, train, and predict with:\n\n"
        "```\npython -m src.retail prepare\npython -m src.main train\npython -m src.main predict\n```\n\n"
        "Then refresh this page."
    )


if __name__ == "__main__":
    main()
