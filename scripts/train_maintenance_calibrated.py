import json
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
import joblib

DATA = Path('data/telematics/sim_telematics.csv')
OUT  = Path('outputs/maintenance'); OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path('outputs/models_maintenance'); MODELS.mkdir(parents=True, exist_ok=True)

def add_rollups(df):
    d = df.sort_values(['vehicle_id','ts']).copy()
    for win in [4,16,32,96]:
        d[f'temp_mean_{win}'] = d.groupby('vehicle_id')['engine_temp_c'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'vib_mean_{win}']  = d.groupby('vehicle_id')['vibration'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'spd_mean_{win}']  = d.groupby('vehicle_id')['speed_kmh'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'volt_mean_{win}'] = d.groupby('vehicle_id')['battery_v'].transform(lambda s: s.rolling(win, min_periods=1).mean())
    d['high_temp'] = (d['engine_temp_c']>95).astype(int)
    d['low_volt']  = (d['battery_v']<13.2).astype(int)
    d['rush_hour'] = d['hour'].isin([7,8,9,16,17,18]).astype(int)
    return d

def make_labels(df, horizon_hours=24*7):
    df = df.sort_values(['vehicle_id','ts']).copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['y'] = 0
    horizon = pd.Timedelta(hours=horizon_hours)
    out = []
    for vid, g in df.groupby('vehicle_id'):
        fails = g.loc[g['failure_event']==1,'ts'].tolist()
        marker = np.zeros(len(g), dtype=int)
        for ft in fails:
            idx = g.index[(g['ts'] <= ft) & (g['ts'] >= ft-horizon)]
            marker[np.searchsorted(g.index.values, idx)] = 1
        gg = g.copy(); gg['y'] = marker
        out.append(gg)
    return pd.concat(out).sort_index()

def main():
    df = pd.read_csv(DATA, parse_dates=['ts'])
    df = add_rollups(df)
    df = make_labels(df, horizon_hours=24*7)

    split_ts = df['ts'].quantile(0.8)
    train_all = df[df['ts']<=split_ts].copy()
    test      = df[df['ts']> split_ts].copy()

    # inner split for calibration: last 20% of train_all
    inner_ts = train_all['ts'].quantile(0.8)
    train = train_all[train_all['ts']<=inner_ts].copy()
    cal   = train_all[train_all['ts']> inner_ts].copy()

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    iso_feats = ['engine_temp_c','battery_v','vibration','speed_kmh','temp_mean_16','volt_mean_16','vib_mean_16']
    iso.fit(train[iso_feats][train['y']==0])

    # anomaly score for all splits
    for d in [train, cal, test]:
        d['anomaly_score'] = -iso.decision_function(d[iso_feats])

    clf_feats = iso_feats + ['high_temp','low_volt','rush_hour','temp_mean_96','volt_mean_96','spd_mean_96','vib_mean_96','anomaly_score']

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    rf.fit(train[clf_feats], train['y'])

    # save RF feature importances for the API
    imps = dict(zip(clf_feats, rf.feature_importances_.tolist()))
    (MODELS/'rf_feature_importances.json').write_text(json.dumps(imps), encoding='utf-8')

    # isotonic calibration on the cal set
    calib = CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')
    calib.fit(cal[clf_feats], cal['y'])

    # Evaluate on test
    proba = calib.predict_proba(test[clf_feats])[:,1]
    ap = average_precision_score(test['y'], proba)
    brier = brier_score_loss(test['y'], proba)

    Path(OUT/'metrics_calibrated.txt').write_text(f'AveragePrecision={ap:.4f}\nBrierScore={brier:.4f}\n', encoding='utf-8')

    # Save artifacts for serving
    joblib.dump(iso, MODELS/'iso_model.joblib')
    joblib.dump(calib, MODELS/'rf_calibrated.joblib')
    (MODELS/'features.txt').write_text('\n'.join(clf_feats), encoding='utf-8')
    print(f'Saved models to {MODELS}. AP={ap:.4f}, Brier={brier:.4f}')

if __name__ == '__main__':
    main()
