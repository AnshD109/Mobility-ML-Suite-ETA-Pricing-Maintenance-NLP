import streamlit as st
import requests

st.set_page_config(page_title="Quote Tool", layout="centered")
st.title("ETA + Price Quote Tool")

st.caption("Point this at your running API (Step 6). Default is http://127.0.0.1:8000")
api_url = st.text_input("API base URL", "http://127.0.0.1:8000")

with st.form("quote_form"):
    col1, col2 = st.columns(2)
    distance_km = col1.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)
    hour = col1.number_input("Hour (0-23)", min_value=0, max_value=23, value=18, step=1)
    day_of_week = col1.number_input("Day of Week (0=Mon..6=Sun)", min_value=0, max_value=6, value=4, step=1)
    demand_index = col2.number_input("Demand index", min_value=0.1, value=1.2, step=0.1)
    supply_index = col2.number_input("Supply index", min_value=0.1, value=0.9, step=0.1)
    city = col2.selectbox("City", ["Berlin","Munich","Other"])
    submitted = st.form_submit_button("Get Quote")

if submitted:
    body = {
        "distance_km": distance_km,
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "demand_index": float(demand_index),
        "supply_index": float(supply_index),
        "city": city,
    }
    try:
        r = requests.post(api_url.rstrip('/') + "/quote", json=body, timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success("Quote computed")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ETA (median)", f"{data['eta_minutes']['p50']:.2f} min")
                st.metric("ETA range (P10–P90)", f"{data['eta_minutes']['p10']:.1f} – {data['eta_minutes']['p90']:.1f} min")
            with c2:
                st.metric("Price", f"€ {data['price']:.2f}")
                st.json(data["price_components"])
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
