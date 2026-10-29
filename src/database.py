from __future__ import annotations

from datetime import datetime, timezone

from supabase import create_client

from src.settings import SUPABASE_KEY, SUPABASE_URL


def save_predictions_to_supabase(predictions: list[dict[str, object]]) -> int:
    client = _get_client()
    predicted_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {"prediction": item["prediction"], "predicted_at": predicted_at} for item in predictions
    ]
    response = client.table("predictions").insert(rows).execute()
    return len(response.data)


def load_predictions_from_supabase(
    credentials: tuple[str, str] | None = None,
) -> list[dict[str, object]]:
    client = _get_client(credentials)
    response = client.table("predictions").select("*").order("predicted_at", desc=True).execute()
    return response.data


def _get_client(credentials: tuple[str, str] | None = None):
    url, key = credentials or (SUPABASE_URL, SUPABASE_KEY)
    if not url or not key:
        raise ValueError("Supabase credentials not configured.")
    return create_client(url, key)
