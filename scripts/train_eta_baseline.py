import pandas as pd
from pathlib import Path
from src.features.eta_features import basic_eta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = Path("data/raw/orders_sample.csv")

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["ts"])
    # Create features
    X = basic_eta_features(df)[[
        "distance_km","dist_sq","hour","day_of_week","peak_hour",
        "demand_index","supply_index","demand_supply_ratio","weekend",
        "city__Berlin","city__Munich"
    ]]
    y = df["eta_actual_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("ETA baseline (RandomForest)")
    print(f"Test MAE (minutes): {mae:.2f}")
    print(f"Test R^2: {r2:.3f}")
    # quick feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:")
    print(importances.head(8))

    # Save quick results
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "eta_baseline_metrics.txt").write_text(f"MAE={mae:.3f}\nR2={r2:.3f}\n")
    importances.to_csv(out_dir / "eta_feature_importances.csv")

if __name__ == "__main__":
    main()
