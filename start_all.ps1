# ETA + Pricing
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; .\.venv\Scripts\Activate.ps1; python -m uvicorn api.eta_service_pricing:app --reload --port 8000"
# Maintenance
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; .\.venv\Scripts\Activate.ps1; python -m uvicorn api.maintenance_service:app --reload --port 8001"
# NLP
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; .\.venv\Scripts\Activate.ps1; python -m uvicorn api.nlp_service:app --reload --port 8002"
