import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from src.features.eta_features import basic_eta_features
except ModuleNotFoundError as e:
    raise SystemExit("Import error: src.features not found. Ensure __init__.py exists in src/ and src/features/. "
                     "Run as 'python -m scripts.train_eta_quantiles' from project root.") from e

def fit_gbr_quantile(X, y, alpha):
    return GradientBoostingRegressor(
        loss='quantile', alpha=alpha,
        n_estimators=400, learning_rate=0.03, max_depth=3, random_state=42
    ).fit(X, y)

def main():
    parser = argparse.ArgumentParser(description="Train ETA quantile models for uncertainty intervals")
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

    feats = [
        "distance_km","dist_sq","hour","day_of_week","peak_hour",
        "demand_index","supply_index","demand_supply_ratio","weekend",
        "city__Berlin","city__Munich"
    ]

    X_train = basic_eta_features(df_train)[feats]
    y_train = df_train["eta_actual_min"]
    X_test  = basic_eta_features(df_test)[feats]
    y_test  = df_test["eta_actual_min"]

    # Train three quantile models
    model_p10 = fit_gbr_quantile(X_train, y_train, alpha=0.10)
    model_p50 = fit_gbr_quantile(X_train, y_train, alpha=0.50)
    model_p90 = fit_gbr_quantile(X_train, y_train, alpha=0.90)

    p10 = model_p10.predict(X_test)
    p50 = model_p50.predict(X_test)
    p90 = model_p90.predict(X_test)

    mae50 = mean_absolute_error(y_test, p50)
    r2_50 = r2_score(y_test, p50)
    coverage90 = np.mean((y_test.values >= p10) & (y_test.values <= p90))
    avg_width = float(np.mean(p90 - p10))

    # Save metrics
    metrics_txt = f"""ETA Quantiles
MAE_P50={mae50:.3f}
R2_P50={r2_50:.3f}
COVERAGE_P10_P90={coverage90:.3f}
AVG_INTERVAL_WIDTH={avg_width:.3f} minutes
"""
    (OUT / 'eta_quantiles_metrics.txt').write_text(metrics_txt, encoding='utf-8')

    # Save predictions
    preds = pd.DataFrame({
        'ts': df_test['ts'].values,
        'actual': y_test.values,
        'pred_p10': p10,
        'pred_p50': p50,
        'pred_p90': p90,
        'hour': df_test['hour'].values,
        'distance_km': df_test['distance_km'].values
    })
    preds.to_csv(OUT / 'preds_quantiles.csv', index=False)

    # Simple plot: actual vs median with interval band
    order = np.arange(len(preds))
    plt.figure(figsize=(10,4))
    plt.plot(order, preds['actual'], label='actual')
    plt.plot(order, preds['pred_p50'], label='pred_p50')
    plt.fill_between(order, preds['pred_p10'], preds['pred_p90'], alpha=0.2, label='P10–P90')
    plt.title('ETA: actual vs predicted median with P10–P90 interval')
    plt.xlabel('Test sample index (time order)')
    plt.ylabel('Minutes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'eta_quantile_plot.png', dpi=120)

    print(metrics_txt)
    print(f"Saved: {OUT/'preds_quantiles.csv'} and {OUT/'eta_quantile_plot.png'}")


if __name__ == '__main__':
    main()
