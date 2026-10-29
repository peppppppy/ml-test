import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_PATH = ROOT_DIR / "data" / "train.csv"
INFERENCE_DATA_PATH = ROOT_DIR / "data" / "inference.csv"
TARGET_COLUMN = "target"
FEATURE_COLUMNS: list[str] = []
MODEL_PATH = ROOT_DIR / "data" / "model.joblib"
PREDICTIONS_PATH = ROOT_DIR / "data" / "predictions.json"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")


def get_supabase_credentials(
    secrets: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    url = SUPABASE_URL or (secrets.get("SUPABASE_URL", "") if secrets is not None else "")
    key = SUPABASE_KEY or (secrets.get("SUPABASE_KEY", "") if secrets is not None else "")
    return url, key
