from pathlib import Path
import pandas as pd

def append_csv_row(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, mode='w', index=False, header=True)
