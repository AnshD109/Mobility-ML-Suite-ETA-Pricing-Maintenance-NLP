# AutoFleet AI — Step 1 (Beginner Starter)

Your **first step** is to set up Python, run a tiny ETA baseline on a sample dataset, and see your first metric.

## What you'll do now
1) Create a virtual environment
2) Install requirements
3) Run the ETA baseline training script
4) Read the MAE (minutes) and top features

## Setup (Windows PowerShell)
```powershell
cd autofleet_ai_step1_starter
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python scripts/train_eta_baseline.py
```

## Setup (macOS/Linux)
```bash
cd autofleet_ai_step1_starter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_eta_baseline.py
```

## Expected output
You'll see something like:
```
ETA baseline (RandomForest)
Test MAE (minutes): ~3–6
Test R^2: ~0.70–0.85
Top features:
...
```
(Your exact numbers will vary because the sample data is synthetic.)

## What's in this starter?
- `data/raw/orders_sample.csv` — synthetic ride orders (Berlin & Munich) so you can train right away.
- `scripts/train_eta_baseline.py` — trains a RandomForest ETA model and prints metrics.
- `src/features/eta_features.py` — simple feature builder for ETA (distance, time, demand/supply).

## Next (after Step 1 succeeds)
- **Step 2:** Turn this into a proper project skeleton (dbt, data quality checks, MLflow, Streamlit).
- **Step 3:** Add modules for Predictive Maintenance and NLP (we'll guide you).

> Tip: Open `scripts/train_eta_baseline.py` and read the code. Try adding/removing a feature and see how MAE changes.
