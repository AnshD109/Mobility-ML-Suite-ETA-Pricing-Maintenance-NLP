
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score

DATA = Path('data/telematics/sim_telematics.csv')
OUT = Path('outputs/maintenance'); OUT.mkdir(parents=True, exist_ok=True)

def add_rollups(df):
    d = df.sort_values(['vehicle_id','ts']).copy()
    for win in [4, 16, 32]:  # ~1h, 4h, 8h
        d[f'temp_mean_{win}'] = d.groupby('vehicle_id')['engine_temp_c'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'vib_mean_{win}']  = d.groupby('vehicle_id')['vibration'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'spd_mean_{win}']  = d.groupby('vehicle_id')['speed_kmh'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        d[f'volt_mean_{win}'] = d.groupby('vehicle_id')['battery_v'].transform(lambda s: s.rolling(win, min_periods=1).mean())
    d['high_temp'] = (d['engine_temp_c']>95).astype(int)
    d['low_volt']  = (d['battery_v']<13.2).astype(int)
    d['rush_hour'] = d['hour'].isin([7,8,9,16,17,18]).astype(int)
    return d

def make_labels(df, horizon_hours=72):
    df = df.sort_values(['vehicle_id','ts']).copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['y'] = 0
    horizon = pd.Timedelta(hours=horizon_hours)
    out = []
    for vid, g in df.groupby('vehicle_id'):
        fails = g.loc[g['failure_event']==1, 'ts'].tolist()
        marker = np.zeros(len(g), dtype=int)
        for ft in fails:
            idx = g.index[(g['ts'] <= ft) & (g['ts'] >= ft - horizon)]
            marker[np.searchsorted(g.index.values, idx)] = 1
        gg = g.copy(); gg['y'] = marker
        out.append(gg)
    return pd.concat(out).sort_index()

def main():
    df = pd.read_csv(DATA, parse_dates=['ts'])
    df = add_rollups(df)
    df = make_labels(df, horizon_hours=72)

    split_ts = df['ts'].quantile(0.8)
    train = df[df['ts']<=split_ts].copy()
    test  = df[df['ts']> split_ts].copy()

    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    iso_feats = ['engine_temp_c','battery_v','vibration','speed_kmh','temp_mean_16','volt_mean_16','vib_mean_16']
    iso.fit(train[iso_feats][train['y']==0])
    train['anomaly_score'] = -iso.decision_function(train[iso_feats])
    test['anomaly_score'] = -iso.decision_function(test[iso_feats])

    clf_feats = iso_feats + ['high_temp','low_volt','rush_hour','temp_mean_32','volt_mean_32','spd_mean_32','vib_mean_32','anomaly_score']
    Xtr = train[clf_feats]; ytr = train['y']
    Xte = test[clf_feats];  yte = test['y']

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample')
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    ap = average_precision_score(yte, proba)

    N = min(50, len(proba))
    topk_idx = np.argsort(proba)[::-1][:N]
    prec_at_k = yte.iloc[topk_idx].mean()

    Path(OUT/'metrics.txt').write_text(f'AveragePrecision={ap:.4f}\nPrecision@{N}={prec_at_k:.4f}\n')
    pd.DataFrame({'ts': test['ts'].values, 'vehicle_id': test['vehicle_id'].values, 'risk': proba}).to_csv(OUT/'test_scores.csv', index=False)
    print(f"Average Precision: {ap:.4f} | Precision@{N}: {prec_at_k:.4f}")
    print(f"Wrote {OUT/'test_scores.csv'}")

if __name__ == '__main__':
    main()
