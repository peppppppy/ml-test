from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import load_predictions_from_supabase
from src.settings import PREDICTIONS_PATH, SUPABASE_KEY, SUPABASE_URL


@st.cache_data
def load_local_predictions(path: str | Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text())
    return pd.DataFrame(data)


st.title("ML Project Dashboard")

if SUPABASE_URL and SUPABASE_KEY:
    predictions = pd.DataFrame(load_predictions_from_supabase())
else:
    predictions = load_local_predictions(PREDICTIONS_PATH)
if predictions.empty:
    st.warning("No predictions yet. Run `make predict` to generate them.")
    st.stop()

fig = px.histogram(predictions, x="prediction", nbins=20)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(predictions.head(20), use_container_width=True)
