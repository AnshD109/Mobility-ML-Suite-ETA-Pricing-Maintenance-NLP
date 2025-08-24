
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
import uuid
import datetime as dt

from src.features.eta_features import basic_eta_features

def ensure_feature_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]

LOG_DIR = Path('outputs') / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'requests.csv'

def append_log(row: dict):
    df = pd.DataFrame([row])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(LOG_FILE, mode='w', index=False, header=True)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'outputs' / 'models'
Q10 = joblib.load(MODELS/'eta_q10.joblib')
Q50 = joblib.load(MODELS/'eta_q50.joblib')
Q90 = joblib.load(MODELS/'eta_q90.joblib')
FEATS = (MODELS/'features.txt').read_text().strip().splitlines()

app = FastAPI(title="ETA Quantiles API (Logged)", version="0.2")

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
    req_id = str(uuid.uuid4())
    ts = dt.datetime.utcnow().isoformat()

    df = pd.DataFrame([item.dict()])
    X = basic_eta_features(df)
    X = ensure_feature_columns(X, FEATS)

    p10 = float(Q10.predict(X)[0])
    p50 = float(Q50.predict(X)[0])
    p90 = float(Q90.predict(X)[0])

    log_row = {
        "req_id": req_id,
        "ts_utc": ts,
        **item.dict(),
        "pred_p10": round(p10, 4),
        "pred_p50": round(p50, 4),
        "pred_p90": round(p90, 4)
    }
    append_log(log_row)

    return {"eta_minutes": {"p10": round(p10,2), "p50": round(p50,2), "p90": round(p90,2)}}
