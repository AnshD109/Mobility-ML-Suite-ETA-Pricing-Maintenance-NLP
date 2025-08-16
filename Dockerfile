FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

# system deps for numpy/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends         build-essential gcc g++     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt ./requirements-api.txt
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# copy project
COPY . /app

# default to ETA + Pricing; override in compose/cmd
CMD ["uvicorn", "api.eta_service_pricing:app", "--host", "0.0.0.0", "--port", "8000"]
