import streamlit as st
import requests

st.set_page_config(page_title="Incident NLP Triage", layout="centered")
st.title("NLP Triage — classify & explore topics")

api_url = st.text_input("API base URL", "http://127.0.0.1:8000")

text = st.text_area("Paste an incident / feedback:", "Driver was speeding and the ETA kept changing; ended up 20 minutes late.")
if st.button("Classify"):
    try:
        r = requests.post(api_url.rstrip("/") + "/nlp/predict", json={"text": text}, timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

if st.button("Show discovered topics"):
    try:
        r = requests.get(api_url.rstrip("/") + "/nlp/topics", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))
