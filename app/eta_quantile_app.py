
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="ETA Uncertainty", layout="wide")
st.title("ETA Quantiles — Median & P10–P90 Interval")

OUT = Path('outputs')
preds_path = OUT / 'preds_quantiles.csv'

if not preds_path.exists():
    st.warning("Run the training first: `python -m scripts.train_eta_quantiles`")
else:
    df = pd.read_csv(preds_path, parse_dates=['ts'])
    df['in_interval'] = (df['actual'] >= df['pred_p10']) & (df['actual'] <= df['pred_p90'])
    mae = (df['actual'] - df['pred_p50']).abs().mean()
    coverage = df['in_interval'].mean()

    c1, c2 = st.columns(2)
    c1.metric("MAE (minutes, using P50)", f"{mae:.2f}")
    c2.metric("Coverage (within P10–P90)", f"{coverage:.1%}")

    st.line_chart(df[['actual','pred_p50']])
    st.area_chart(df[['pred_p10','pred_p90']])

    st.subheader("Preview of predictions")
    st.dataframe(df.head(50))
