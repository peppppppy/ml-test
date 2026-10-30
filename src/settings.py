"""Configuration — the project's control panel.

Anything you'd tweak without touching pipeline logic lives here.
"""

from __future__ import annotations

# ── Data ──────────────────────────────────────────────────────────────────────
# Retail source data and generated datasets used by the demo.
RETAIL_DATA_PATH: str = "data/retail_sales.csv"
DATA_PATH: str = "data/review_score_train.csv"
INFERENCE_DATA_PATH: str = "data/review_score_inference.csv"

# ── Features & target ─────────────────────────────────────────────────────────
TARGET_COLUMN: str = "review_score"
FEATURE_COLUMNS: list[str] = [
    "price",
    "quantity",
    "category_name",
    "payment_method",
    "age",
]

# ── Train / validation split ──────────────────────────────────────────────────
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ── MLflow experiment tracking ────────────────────────────────────────────────
# Set MLFLOW_TRACKING_URI to a remote URI (e.g. "http://mlflow:5000") for a
# shared server. The default writes runs locally to mlruns/.
MLFLOW_EXPERIMENT_NAME: str = "ml-project-starter"
MLFLOW_TRACKING_URI: str = "mlruns"

# ── Storage ───────────────────────────────────────────────────────────────────
SUPABASE_TABLE_NAME: str = "review_score_predictions"
SUPABASE_RETAIL_TABLE_NAME: str = "retail_sales"
LOCAL_FALLBACK_PATH: str = "data/review_score_predictions.json"

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE: str = "Online Retail Review Score Dashboard"
DASHBOARD_SUBTITLE: str = (
    "Explore retail data and review-score predictions."
)
