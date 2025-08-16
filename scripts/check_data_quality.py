import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Data quality checks")
    parser.add_argument('--data', default=None, help="Path to orders_sample.csv")
    parser.add_argument('--out', default=None, help="Output folder")
    args = parser.parse_args()

    # Resolve project root relative to this file (so it works from any cwd)
    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data) if args.data else ROOT / 'data' / 'raw' / 'orders_sample.csv'
    OUT = Path(args.out) if args.out else ROOT / 'outputs'
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(f"Could not find data file at: {DATA}")

    df = pd.read_csv(DATA, parse_dates=['ts'])
    issues = []

    summary = []
    summary.append(f'Rows: {len(df)}')
    summary.append(f'Columns: {list(df.columns)}')

    must_have = ['order_id','ts','distance_km','hour','day_of_week','eta_promised_min','eta_actual_min']
    nulls = df[must_have].isnull().sum()
    for c, n in nulls.items():
        if n>0: issues.append(f'NULLS: column {c} has {n} nulls')

    dups = df['order_id'].duplicated().sum()
    if dups>0: issues.append(f'DUPLICATES: order_id has {dups} duplicates')

    if (df['distance_km']<0).any(): issues.append('distance_km has negative values')
    if (df['distance_km']>80).any(): issues.append('distance_km has unrealistically large values (>80km)')
    if (~df['hour'].between(0,23)).any(): issues.append('hour outside 0..23')
    if (~df['day_of_week'].between(0,6)).any(): issues.append('day_of_week outside 0..6')
    if (df['eta_promised_min']<0).any() or (df['eta_actual_min']<0).any():
        issues.append('ETA columns contain negatives')

    for c in ['demand_index','supply_index']:
        if (df[c]<=0).any(): issues.append(f'{c} contains non-positive values')

    report = ['DATA QUALITY REPORT'] + summary + ['','Issues:']
    if issues:
        report += [f' - {x}' for x in issues]
    else:
        report += [' - None found']

    (OUT/'data_quality_report.txt').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))

if __name__ == '__main__':
    main()
