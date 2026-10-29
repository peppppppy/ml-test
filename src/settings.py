import os
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
