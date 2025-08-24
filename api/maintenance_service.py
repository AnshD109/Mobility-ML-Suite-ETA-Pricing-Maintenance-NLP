from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import json

# pydantic v1-compatible models
class TelemetryPoint(BaseModel):
    ts: Optional[str] = None
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    speed_kmh: float
    engine_temp_c: float
    battery_v: float
    vibration: float

class RiskRequest(BaseModel):
    vehicle_id: str
    history: List[TelemetryPoint]  # send last 32 points (15-min each)

app = FastAPI(title="Maintenance Risk API", version="0.1")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'outputs' / 'models_maintenance'
ISO = joblib.load(MODELS/'iso_model.joblib')
CAL = joblib.load(MODELS/'rf_calibrated.joblib')
FEATS = (MODELS/'features.txt').read_text().splitlines()
IMP_PATH = MODELS / 'rf_feature_importances.json'
IMPS = json.loads(IMP_PATH.read_text()) if IMP_PATH.exists() else {}

def build_features(history: pd.DataFrame) -> pd.DataFrame:
    # Copy & order by time if a timestamp is present
    df = history.copy()
    if 'ts' in df.columns:
        df = df.sort_values('ts')

    # Rolling features
    for win in [4, 16, 32, 96]:
        df[f'temp_mean_{win}'] = df['engine_temp_c'].rolling(win, min_periods=1).mean()
        df[f'vib_mean_{win}']  = df['vibration'].rolling(win, min_periods=1).mean()
        df[f'spd_mean_{win}']  = df['speed_kmh'].rolling(win, min_periods=1).mean()
        df[f'volt_mean_{win}'] = df['battery_v'].rolling(win, min_periods=1).mean()

    df['high_temp'] = (df['engine_temp_c'] > 95).astype(int)
    df['low_volt']  = (df['battery_v'] < 13.2).astype(int)
    df['rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # On-the-fly anomaly score (same features used during training)
    iso_feats = ['engine_temp_c','battery_v','vibration','speed_kmh',
                 'temp_mean_16','volt_mean_16','vib_mean_16']
    df['anomaly_score'] = -ISO.decision_function(df[iso_feats])

    return df


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/maintenance/risk")
def maintenance_risk(req: RiskRequest):
    # convert history to DataFrame
    hist = pd.DataFrame([h.dict() for h in req.history])
    if 'ts' in hist.columns and hist['ts'].notna().any():
        hist['ts'] = pd.to_datetime(hist['ts'])
        hist = hist.sort_values('ts')
    else:
        hist.reset_index(drop=True, inplace=True)
        hist['ts'] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(hist), freq='15min')
    feats_df = build_features(hist)
    x = feats_df.iloc[[-1]][FEATS]  # take last row and select expected feature columns
    risk = float(CAL.predict_proba(x)[:,1][0])
    try:
        # get feature importances from underlying RF
        rf = CAL.base_estimator
        importances = dict(sorted(zip(FEATS, rf.feature_importances_), key=lambda t: t[1], reverse=True)[:8])
    except Exception:
        importances = {}
    top = dict(sorted(IMPS.items(), key=lambda t: t[1], reverse=True)[:8])
    return {"vehicle_id": req.vehicle_id, "risk": round(risk, 4), "top_features": top}

