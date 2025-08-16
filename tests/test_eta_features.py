import pandas as pd
from src.features.eta_features import basic_eta_features

def test_basic_eta_features_shapes():
    df = pd.DataFrame([{
        "distance_km": 5.0,
        "hour": 18,
        "day_of_week": 4,
        "demand_index": 1.2,
        "supply_index": 0.9,
        "city": "Berlin"
    }])
    X = basic_eta_features(df)
    assert not X.empty
    assert "distance_km" in X.columns
