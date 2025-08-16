from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd

from src.features.eta_features import basic_eta_features
from policy.pricing_rules import compute_price

def ensure_feature_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'outputs' / 'models'
Q10 = joblib.load(MODELS/'eta_q10.joblib')
Q50 = joblib.load(MODELS/'eta_q50.joblib')
Q90 = joblib.load(MODELS/'eta_q90.joblib')
FEATS = (MODELS/'features.txt').read_text().strip().splitlines()

app = FastAPI(title="ETA + Pricing API", version="0.1")

class RideIn(BaseModel):
    distance_km: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    demand_index: float = Field(..., gt=0)
    supply_index: float = Field(..., gt=0)
    city: str = Field(..., description="e.g., Berlin or Munich")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: RideIn):
    df = pd.DataFrame([item.dict()])
    X = ensure_feature_columns(basic_eta_features(df), FEATS)
    p10 = float(Q10.predict(X)[0]); p50 = float(Q50.predict(X)[0]); p90 = float(Q90.predict(X)[0])
    return {"eta_minutes": {"p10": round(p10,2), "p50": round(p50,2), "p90": round(p90,2)}}

@app.post("/quote")
def quote(item: RideIn):
    df = pd.DataFrame([item.dict()])
    X = ensure_feature_columns(basic_eta_features(df), FEATS)
    p10 = float(Q10.predict(X)[0]); p50 = float(Q50.predict(X)[0]); p90 = float(Q90.predict(X)[0])
    width = max(0.0, p90 - p10)
    pricing = compute_price(item.distance_km, p50, item.demand_index, item.supply_index, width)
    return {
        "eta_minutes": {"p10": round(p10,2), "p50": round(p50,2), "p90": round(p90,2), "width": round(width,2)},
        "price": pricing["price"],
        "price_components": pricing["components"]
    }
