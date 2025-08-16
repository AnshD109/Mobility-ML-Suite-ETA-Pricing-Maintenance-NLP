import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    from src.features.eta_features import basic_eta_features
except ModuleNotFoundError as e:
    raise SystemExit("Import error: src.features not found. Ensure __init__.py exists in src/ and src/features/. "
                     "Run from project root.") from e

def fit_gbr_quantile(X, y, alpha):
    return GradientBoostingRegressor(
        loss='quantile', alpha=alpha,
        n_estimators=400, learning_rate=0.03, max_depth=3, random_state=42
    ).fit(X, y)

def main():
    parser = argparse.ArgumentParser(description="Train ETA quantiles and save models to disk")
    parser.add_argument('--data', default=None, help="Path to orders_sample.csv")
    parser.add_argument('--out', default=None, help="Output folder")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data) if args.data else ROOT / 'data' / 'raw' / 'orders_sample.csv'
    OUT = Path(args.out) if args.out else ROOT / 'outputs' / 'models'
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(f"Could not find data file at: {DATA}")

    df = pd.read_csv(DATA, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    split_idx = int(len(df)*0.8)
    train = df.iloc[:split_idx].copy()

    feats = [
        "distance_km","dist_sq","hour","day_of_week","peak_hour",
        "demand_index","supply_index","demand_supply_ratio","weekend",
        "city__Berlin","city__Munich"
    ]

    X = basic_eta_features(train)[feats]
    y = train["eta_actual_min"]

    m10 = fit_gbr_quantile(X, y, 0.10)
    m50 = fit_gbr_quantile(X, y, 0.50)
    m90 = fit_gbr_quantile(X, y, 0.90)

    joblib.dump(m10, OUT/'eta_q10.joblib')
    joblib.dump(m50, OUT/'eta_q50.joblib')
    joblib.dump(m90, OUT/'eta_q90.joblib')

    (OUT/'features.txt').write_text("\n".join(feats), encoding='utf-8')
    print(f"Saved models to {OUT}")

if __name__ == '__main__':
    main()
