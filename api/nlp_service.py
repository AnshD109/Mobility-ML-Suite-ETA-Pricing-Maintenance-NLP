from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import joblib, json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models_nlp"

VEC = joblib.load(MODELS/"tfidf.joblib")
CLF = joblib.load(MODELS/"clf_lr.joblib")
LE  = joblib.load(MODELS/"label_encoder.joblib")
TOPICS = json.loads((MODELS/"topics.json").read_text()) if (MODELS/"topics.json").exists() else {}

app = FastAPI(title="Incident NLP Triage API", version="0.1")

class TextIn(BaseModel):
    text: str = Field(..., description="Incident/feedback text")

class BatchIn(BaseModel):
    items: List[TextIn]

def explain_terms(text: str, top_k: int = 6):
    X = VEC.transform([text.lower()])
    row = X.tocoo()
    weights = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)[:top_k]
    feats = VEC.get_feature_names_out()
    return [{"term": feats[i], "weight": float(w)} for i, w in weights]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/nlp/predict")
def predict(inp: TextIn):
    x = VEC.transform([inp.text.lower()])
    pred = int(CLF.predict(x)[0])
    proba = float(CLF.predict_proba(x)[0][pred]) if hasattr(CLF, "predict_proba") else None
    label = LE.inverse_transform([pred])[0]
    return {"label": label, "confidence": round(proba, 4) if proba is not None else None, "explain_terms": explain_terms(inp.text)}

@app.post("/nlp/batch_predict")
def batch_predict(inp: BatchIn):
    texts = [it.text.lower() for it in inp.items]
    x = VEC.transform(texts)
    preds = CLF.predict(x)
    labels = LE.inverse_transform(preds)
    return {"labels": labels.tolist()}

@app.get("/nlp/topics")
def list_topics():
    return {"topics": TOPICS}
