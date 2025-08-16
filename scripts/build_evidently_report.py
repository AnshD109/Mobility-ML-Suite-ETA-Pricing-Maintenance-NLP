import argparse
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None, help='Reference CSV (training-like)')
    parser.add_argument('--cur', default=None, help='Current CSV (API logs)')
    parser.add_argument('--out', default=None, help='Output HTML file')
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    REF = Path(args.ref) if args.ref else ROOT/'data'/'raw'/'orders_sample.csv'
    CUR = Path(args.cur) if args.cur else ROOT/'outputs'/'logs'/'requests.csv'
    OUT = Path(args.out) if args.out else ROOT/'outputs'/'monitoring'/'evidently_report.html'
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not REF.exists():
        raise FileNotFoundError(f"Reference not found: {REF}")
    if not CUR.exists():
        raise FileNotFoundError(f"Current not found: {CUR}")

    ref = pd.read_csv(REF, parse_dates=['ts'])
    cur = pd.read_csv(CUR, parse_dates=['ts_utc'])

    use_cols = ['distance_km','hour','day_of_week','demand_index','supply_index','city']
    ref_use = ref[use_cols].copy()
    cur_use = cur[use_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_use, current_data=cur_use)
    report.save_html(str(OUT))
    print(f"Saved Evidently drift report to: {OUT}")

if __name__ == '__main__':
    main()
