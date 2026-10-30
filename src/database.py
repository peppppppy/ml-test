"""Storage layer.

Writes predictions to a database so the dashboard can read them back. The default
backend is Supabase (hosted PostgreSQL with a generous free tier). If Supabase
isn't configured the pipeline falls back to a local JSON file so you can run
the demo end-to-end without any external dependencies.

ENV VARS REQUIRED FOR SUPABASE:
    SUPABASE_URL — your project URL (https://<project-ref>.supabase.co)
    SUPABASE_KEY — service-role key (NOT the anon key; needed for writes)

The schema of the Supabase table is deliberately open-ended: create a table
named via settings.SUPABASE_TABLE_NAME with whatever columns match the dicts
produced by output.format_predictions(). A "prediction" (float8) column and a
"predicted_at" (timestamptz) column are always present; add more to match your
use case.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)


def save_predictions(rows: list[dict]) -> None:
    """Persist a batch of predictions.

    Args:
        rows: list of dicts as returned by output.format_predictions().
              Each dict becomes one row in storage.
    """
    if _supabase_configured():
        _save_to_supabase(rows)
    else:
        logger.warning(
            "SUPABASE_URL/SUPABASE_KEY not set — writing to local JSON at %s. "
            "Configure Supabase before deploying (see README).",
            settings.LOCAL_FALLBACK_PATH,
        )
        _save_to_local_file(rows)


def load_recent_predictions(limit: int = 100) -> list[dict]:
    """Load the most recent N predictions for the dashboard to display."""
    if _supabase_configured():
        return _load_from_supabase(limit)
    return _load_from_local_file(limit)


def save_retail_sales(rows: list[dict]) -> None:
    """Upsert source retail rows, keyed by their stable source-row number."""
    if not _supabase_configured():
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY are required to import retail rows."
        )

    client = _get_supabase_client()
    for start in range(0, len(rows), 500):
        batch = rows[start : start + 500]
        client.table(settings.SUPABASE_RETAIL_TABLE_NAME).upsert(
            batch, on_conflict="source_row"
        ).execute()
    logger.info(
        "Upserted %d rows into Supabase table %r",
        len(rows),
        settings.SUPABASE_RETAIL_TABLE_NAME,
    )


def load_retail_sales(limit: int = 5000) -> list[dict]:
    """Load retail rows from Supabase, or the source CSV for local preview."""
    if _supabase_configured():
        client = _get_supabase_client()
        response = (
            client.table(settings.SUPABASE_RETAIL_TABLE_NAME)
            .select("*")
            .order("source_row")
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            row.setdefault(settings.RETURN_TARGET_COLUMN, None)
        return rows

    path = Path(settings.RETAIL_DATA_PATH)
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit).copy()
    df.insert(0, "source_row", range(1, len(df) + 1))
    if settings.RETURN_TARGET_COLUMN not in df.columns:
        df[settings.RETURN_TARGET_COLUMN] = pd.NA
    df = df.loc[
        :,
        [
            "source_row",
            *settings.FEATURE_COLUMNS,
            settings.TARGET_COLUMN,
            settings.RETURN_TARGET_COLUMN,
        ],
    ]
    df["price"] = pd.to_numeric(df["price"], errors="raise").astype(float)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise").astype("int64")
    df["age"] = pd.to_numeric(df["age"], errors="raise").astype("int64")
    df[settings.TARGET_COLUMN] = pd.to_numeric(df[settings.TARGET_COLUMN], errors="coerce")
    returned = pd.to_numeric(df[settings.RETURN_TARGET_COLUMN], errors="raise")
    invalid = returned.notna() & ~returned.isin([0, 1])
    if invalid.any():
        raise ValueError("is_returned values must be 0, 1, or blank.")
    df[settings.RETURN_TARGET_COLUMN] = returned.astype("Int64")
    return json.loads(df.to_json(orient="records", date_format="iso"))


def is_supabase_configured() -> bool:
    """Return whether Supabase credentials are available for this process."""
    return _supabase_configured()


# ------------------------------------------------------------------
# Supabase backend
# ------------------------------------------------------------------

def _supabase_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


def _get_supabase_client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _save_to_supabase(rows: list[dict]) -> None:
    client = _get_supabase_client()
    table = client.table(settings.SUPABASE_TABLE_NAME)
    if rows and all("source_row" in row for row in rows):
        table.upsert(rows, on_conflict="source_row").execute()
    else:
        table.insert(rows).execute()
    logger.info("Inserted %d rows into Supabase table %r", len(rows), settings.SUPABASE_TABLE_NAME)


def _load_from_supabase(limit: int) -> list[dict]:
    client = _get_supabase_client()
    response = (
        client.table(settings.SUPABASE_TABLE_NAME)
        .select("*")
        .order("predicted_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


# ------------------------------------------------------------------
# Local JSON-file backend (fallback for offline development)
# ------------------------------------------------------------------

def _save_to_local_file(rows: list[dict]) -> None:
    path = Path(settings.LOCAL_FALLBACK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = json.loads(path.read_text()) if path.exists() else []
    existing.extend(rows)
    path.write_text(json.dumps(existing, indent=2))
    logger.info("Wrote %d rows to %s", len(rows), path)


def _load_from_local_file(limit: int) -> list[dict]:
    path = Path(settings.LOCAL_FALLBACK_PATH)
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return rows[-limit:]
