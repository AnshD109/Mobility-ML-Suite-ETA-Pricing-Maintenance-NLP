```markdown
# Mobility ML Suite — ETA, Pricing, Maintenance & NLP

End-to-end mobility data-science project that mirrors real work:
- **ETA quantiles** → **pricing** with business rules  
- **Predictive maintenance** from telematics (anomaly + calibrated risk)  
- **NLP incident/feedback triage**  
- **Productionization**: FastAPI services, monitoring with Evidently, and tests

> Tech: Python 3.11 · FastAPI · scikit-learn · pandas · joblib · Evidently · Streamlit  
> OS focus: Windows/PowerShell (commands work cross-platform with small tweaks)

---

## Project structure
```

api/                      # FastAPI apps: eta\_service\_pricing.py, maintenance\_service.py, nlp\_service.py
app/                      # Streamlit mini apps (quote tool, NLP triage)
scripts/                  # Training + utilities + monitoring
src/                      # Feature code (ETA)
outputs/                  # Models, logs, reports (created at runtime)
tests/                    # Tiny sanity tests (pytest)
requirements-api.txt      # Runtime deps (Pydantic v1)
requirements-dev.txt      # Dev/monitoring deps (Evidently, Streamlit, etc.)

````

---

## Environments (two-venv setup)

This repo intentionally uses **two virtual environments** to avoid dependency clashes between FastAPI (Pydantic v1) and Evidently.

### 1) API env — run the services (FastAPI + Pydantic v1)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
````

### 2) Dev/monitoring env — build reports & Streamlit

```powershell
py -3.11 -m venv .venv-dev
.\.venv-dev\Scripts\Activate.ps1
pip install evidently==0.4.15 streamlit==1.36.0 plotly==6.3.0 statsmodels==0.14.5 nltk==3.9.1 `
            pydantic==1.10.13 numpy==1.26.4
```

> Why two envs? Evidently ≥0.4.x pulls newer FastAPI/compat layers that conflict with the API pins. Splitting keeps everything stable.

---

## Train models (one-time or whenever you retrain)

```powershell
# Activate API env first
.\.venv\Scripts\Activate.ps1

# 1) ETA models (+ quantiles) → outputs/models
python -m scripts.train_and_save_eta_models

# 2) Maintenance (IsolationForest + RandomForest + isotonic calibration)
python -m scripts.train_maintenance_calibrated

# 3) NLP TF-IDF + Logistic Regression (+ NMF topics)
python scripts\train_nlp_triage.py
```

> If maintenance AP is \~0 (few positives), open `scripts\make_telematics_data.py`, increase e.g.
> `n_vehicles=60, days=60`, then rerun the two lines above.

---

## Run the services (local)

Open **three terminals** (each with `.venv` activated):

```powershell
# 1) ETA + Pricing on :8000
python -m uvicorn api.eta_service_pricing:app --reload --port 8000
```

```powershell
# 2) Maintenance Risk on :8001
python -m uvicorn api.maintenance_service:app --reload --port 8001
```

```powershell
# 3) NLP Triage on :8002
python -m uvicorn api.nlp_service:app --reload --port 8002
```

Health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
```

Swagger UIs:

* ETA+Pricing → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Maintenance → [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
* NLP → [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

### Example requests

**POST /quote** (8000)

```json
{"distance_km": 6.5, "hour": 18, "day_of_week": 5, "demand_index": 1.4, "supply_index": 0.8, "city": "Berlin"}
```

**POST /maintenance/risk** (8001)

```json
{
  "vehicle_id": "V101",
  "history": [
    {"hour": 17, "day_of_week": 4, "speed_kmh": 32, "engine_temp_c": 99.5, "battery_v": 13.1, "vibration": 0.32},
    {"hour": 18, "day_of_week": 4, "speed_kmh": 35, "engine_temp_c": 103.0, "battery_v": 13.0, "vibration": 0.38}
  ]
}
```

**POST /nlp/predict** (8002)

```json
{"text":"App kept crashing at checkout and the price looked wrong"}
```

---

## Monitoring & drift (dev env)

```powershell
# Optional: generate request logs first
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.eta_service_logged:app --reload
# (exercise the API, then Ctrl+C)

# Build the report in dev env
.\.venv-dev\Scripts\Activate.ps1
python scripts\build_evidently_report.py
start .\outputs\monitoring\evidently_report.html
```

Add 1–2 screenshots of the report to your repo.

---

## Streamlit mini apps (optional)

```powershell
# Quote tool (if included)
streamlit run app\quote_tool.py

# NLP triage explorer
streamlit run app\nlp_triage_app.py
```

---

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
pip install pytest
python -m pytest -q
```

---

## Results (fill with your numbers)

* **ETA**: MAE ≈ *X* min, R² ≈ *Y*
* **Maintenance**: Average Precision ≈ *A* (see `outputs/maintenance/metrics_calibrated.txt`)
* **NLP**: F1\_macro ≈ *B* (printed by the training script)

---

## Troubleshooting

* **Port already in use** → stop old servers (Ctrl+C) or change `--port`.
* **Model file not found** → rerun the training commands.
* **Evidently import errors** → use `.venv-dev` with `pydantic==1.10.13`, `numpy==1.26.4`.
* **Pytest imports failing** → ensure you run `python -m pytest` *after* activating `.venv`.

---

## .gitignore (suggested)

```
.venv/
.venv-dev/
__pycache__/
*.pyc
outputs/
*.log
.idea/
.vscode/
.pytest_cache/
```

---

## License

MIT — see [LICENSE](LICENSE) (or choose your preferred license).

## Author

Your Name — LinkedIn • Portfolio

```

::contentReference[oaicite:0]{index=0}
```
