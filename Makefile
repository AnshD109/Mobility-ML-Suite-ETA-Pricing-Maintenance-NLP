.PHONY: help setup train-eta train-maint train-nlp api streamlit docker-build up down test

help:
	@echo "Targets: setup, train-eta, train-maint, train-nlp, api, streamlit, docker-build, up, down, test"

setup:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt

train-eta:
	python -m scripts.train_and_save_eta_models

train-maint:
	python -m scripts.train_maintenance_calibrated

train-nlp:
	python scripts/train_nlp_triage.py

api:
	uvicorn api.eta_service_pricing:app --reload

streamlit:
	streamlit run app/quote_tool.py

docker-build:
	docker build -t autofleet-api:latest .

up:
	docker compose up -d --build

down:
	docker compose down

test:
	pytest -q
