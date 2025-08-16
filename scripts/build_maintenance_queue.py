import pandas as pd
from pathlib import Path

SCORES = Path('outputs/maintenance/test_scores.csv')

def main():
    df = pd.read_csv(SCORES, parse_dates=['ts'])
    latest = df.sort_values('ts').groupby('vehicle_id').tail(1).sort_values('risk', ascending=False)
    top = latest.head(5)
    print("Top vehicles to inspect:")
    print(top[['vehicle_id','ts','risk']])
    top.to_csv('outputs/maintenance/queue_top5.csv', index=False)
    print("Saved outputs/maintenance/queue_top5.csv")

if __name__ == '__main__':
    main()
