import numpy as np
import pandas as pd

def basic_eta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ensure required columns exist
    required = ["distance_km", "hour", "day_of_week", "demand_index", "supply_index", "city"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # non-linear terms & interactions
    out["dist_sq"] = out["distance_km"]**2
    out["peak_hour"] = ((out["hour"].between(7,9)) | (out["hour"].between(16,19))).astype(int)
    out["demand_supply_ratio"] = out["demand_index"]/(out["supply_index"]+1e-6)
    out["weekend"] = out["day_of_week"].isin([5,6]).astype(int)
    # city as one-hot
    for val in out["city"].unique():
        out[f"city__{val}"] = (out["city"]==val).astype(int)
    return out
