from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import load_predictions_from_supabase  # noqa: E402
from src.settings import PREDICTIONS_PATH, get_supabase_credentials  # noqa: E402


@st.cache_data
def load_local_predictions(path: str | Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text())
    return pd.DataFrame(data)


st.title("ML Project Dashboard")

try:
    supabase_credentials = get_supabase_credentials(st.secrets)
except StreamlitSecretNotFoundError:
    supabase_credentials = get_supabase_credentials()

if all(supabase_credentials):
    predictions = pd.DataFrame(load_predictions_from_supabase(supabase_credentials))
else:
    predictions = load_local_predictions(PREDICTIONS_PATH)
if predictions.empty:
    st.warning("No predictions yet. Run `make predict` to generate them.")
    st.stop()

fig = px.histogram(predictions, x="prediction", nbins=20)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(predictions.head(20), use_container_width=True)
