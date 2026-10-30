"""Prepare and import the retail demo data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src import database, settings

load_dotenv()

logger = logging.getLogger(__name__)


def build_retail_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create labeled training rows and inference rows from source retail data."""
    data = df.copy()
    if "source_row" not in data.columns:
        data.insert(0, "source_row", range(1, len(data) + 1))

    data["price"] = pd.to_numeric(data["price"], errors="raise").astype(float)
    data["quantity"] = pd.to_numeric(data["quantity"], errors="raise").astype("int64")
    data["age"] = pd.to_numeric(data["age"], errors="raise").astype("int64")
    data[settings.TARGET_COLUMN] = pd.to_numeric(data[settings.TARGET_COLUMN], errors="coerce")

    training = data.loc[data[settings.TARGET_COLUMN].notna()].copy()
    inference = data.copy()
    return training, inference


def prepare_retail_datasets() -> tuple[Path, Path]:
    """Read the source CSV and write the generated training and inference CSVs."""
    source_path = Path(settings.RETAIL_DATA_PATH)
    if not source_path.exists():
        raise FileNotFoundError(f"Retail source CSV not found: {source_path}")

    training, inference = build_retail_datasets(pd.read_csv(source_path))
    if training.empty:
        raise ValueError("The retail data has no numeric review scores to train on.")

    training_path = Path(settings.DATA_PATH)
    inference_path = Path(settings.INFERENCE_DATA_PATH)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(training_path, index=False)
    inference.to_csv(inference_path, index=False)
    logger.info(
        "Prepared %d labeled rows and %d inference rows",
        len(training),
        len(inference),
    )
    return training_path, inference_path


def import_retail_data() -> int:
    """Upsert the source CSV into the configured Supabase retail table."""
    source_path = Path(settings.RETAIL_DATA_PATH)
    if not source_path.exists():
        raise FileNotFoundError(f"Retail source CSV not found: {source_path}")

    data = pd.read_csv(source_path)
    data.insert(0, "source_row", range(1, len(data) + 1))
    columns = ["source_row", *settings.FEATURE_COLUMNS, settings.TARGET_COLUMN]
    typed_data = data.loc[:, columns].copy()
    typed_data["price"] = pd.to_numeric(typed_data["price"], errors="raise").astype(float)
    typed_data["quantity"] = pd.to_numeric(typed_data["quantity"], errors="raise").astype("int64")
    typed_data["age"] = pd.to_numeric(typed_data["age"], errors="raise").astype("int64")
    typed_data[settings.TARGET_COLUMN] = pd.to_numeric(
        typed_data[settings.TARGET_COLUMN], errors="coerce"
    )
    rows = json.loads(typed_data.to_json(orient="records"))
    database.save_retail_sales(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "import"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "prepare":
        training_path, inference_path = prepare_retail_datasets()
        print(f"Prepared training data: {training_path}")
        print(f"Prepared inference data: {inference_path}")
    else:
        count = import_retail_data()
        print(f"Imported {count} retail rows into Supabase.")


if __name__ == "__main__":
    main()
