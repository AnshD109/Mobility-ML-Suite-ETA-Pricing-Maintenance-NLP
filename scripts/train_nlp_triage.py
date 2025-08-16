
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report, f1_score, accuracy_score

DATA = Path("data/incidents.csv")
OUT_DIR = Path("outputs/nlp"); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS = Path("outputs/models_nlp"); MODELS.mkdir(parents=True, exist_ok=True)

def clean(x: str) -> str:
    return x.lower()

def main():
    if not DATA.exists():
        from scripts.make_incidents_data import generate_csv
        generate_csv(str(DATA), n_rows=300, seed=7)

    df = pd.read_csv(DATA)
    df["text_clean"] = df["text"].fillna("").apply(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    yte = le.transform(y_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    # Topics via NMF
    n_comp = 6
    nmf = NMF(n_components=n_comp, random_state=42, init="nndsvda", max_iter=400)
    nmf.fit(Xtr)

    # Save models
    joblib.dump(vec, MODELS/"tfidf.joblib")
    joblib.dump(clf, MODELS/"clf_lr.joblib")
    joblib.dump(le,  MODELS/"label_encoder.joblib")
    joblib.dump(nmf, MODELS/"nmf_topics.joblib")

    # Save readable topics
    feature_names = np.array(vec.get_feature_names_out())
    top_terms = {}
    for ti, comp in enumerate(nmf.components_):
        idx = np.argsort(comp)[::-1][:8]
        top_terms[int(ti)] = feature_names[idx].tolist()
    (MODELS/"topics.json").write_text(json.dumps(top_terms, indent=2), encoding="utf-8")

    rep = classification_report(yte, ypred, target_names=le.classes_)
    (OUT_DIR/"triage_metrics.txt").write_text(f"Accuracy={acc:.3f}\nF1_macro={f1m:.3f}\n\n{rep}", encoding="utf-8")
    print(f"Accuracy={acc:.3f} | F1_macro={f1m:.3f}")
    print("Saved models to outputs/models_nlp")

if __name__ == "__main__":
    main()
