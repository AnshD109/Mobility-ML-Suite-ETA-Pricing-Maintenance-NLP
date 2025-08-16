import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Import features using package-style path; make sure src has __init__.py
try:
    from src.features.eta_features import basic_eta_features
except ModuleNotFoundError as e:
    raise SystemExit("Cannot import src.features. Make sure you created "
                     "__init__.py in src/ and src/features/. "
                     "Then run as: python -m scripts.train_eta_timeaware") from e

def main():
    parser = argparse.ArgumentParser(description="Time-aware ETA training")
    parser.add_argument('--data', default=None, help="Path to orders_sample.csv")
    parser.add_argument('--out', default=None, help="Output folder")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data) if args.data else ROOT / 'data' / 'raw' / 'orders_sample.csv'
    OUT = Path(args.out) if args.out else ROOT / 'outputs'
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(f"Could not find data file at: {DATA}")

    df = pd.read_csv(DATA, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)

    split_idx = int(len(df)*0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    X_train = basic_eta_features(df_train)[[
        "distance_km","dist_sq","hour","day_of_week","peak_hour",
        "demand_index","supply_index","demand_supply_ratio","weekend",
        "city__Berlin","city__Munich"
    ]]
    y_train = df_train["eta_actual_min"]

    X_test = basic_eta_features(df_test)[[
        "distance_km","dist_sq","hour","day_of_week","peak_hour",
        "demand_index","supply_index","demand_supply_ratio","weekend",
        "city__Berlin","city__Munich"
    ]]
    y_test = df_test["eta_actual_min"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("ETA (Time-aware split)")
    print(f"Test MAE (minutes): {mae:.2f}")
    print(f"Test R^2: {r2:.3f}")

    residuals = (y_test - preds)
    by_hour = pd.DataFrame({'hour': df_test['hour'].values, 'abs_err': (residuals).abs()})\
                .groupby('hour')['abs_err'].mean().rename('mae').reset_index()
    by_hour.to_csv(OUT/'mae_by_hour.csv', index=False)

    plt.figure()
    plt.plot(by_hour['hour'], by_hour['mae'], marker='o')
    plt.title('MAE by Hour (Test set)')
    plt.xlabel('Hour of day')
    plt.ylabel('Mean Absolute Error (minutes)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(OUT/'mae_by_hour.png', bbox_inches='tight')
    (OUT/'eta_timeaware_metrics.txt').write_text(f"MAE={mae:.3f}\nR2={r2:.3f}\n", encoding='utf-8')

if __name__ == '__main__':
    main()
