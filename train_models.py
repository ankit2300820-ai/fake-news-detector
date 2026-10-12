"""Train 4 ML models on the Fake/True news dataset and save results.
Run once:  python train_models.py
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import clean_text

DATA, MODELS = Path("data"), Path("models")
MODELS.mkdir(exist_ok=True)

# 1. Load data (Fake.csv has some broken rows, so keep only title + text) ----
print("Loading dataset...")
fake = pd.read_csv(DATA / "Fake.csv", usecols=["title", "text"], dtype=str, low_memory=False)
real = pd.read_csv(DATA / "True.csv", usecols=["title", "text"], dtype=str, low_memory=False)
fake["label"], real["label"] = 0, 1          # 0 = FAKE, 1 = REAL
df = pd.concat([fake, real], ignore_index=True).dropna(subset=["title", "text"])
df = df.drop_duplicates(subset=["title", "text"])
df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
df = df[df["content"].str.split().str.len() >= 20]   # drop empty / tiny rows
print(f"Rows used: {len(df)}  (fake={int((df.label == 0).sum())}, real={int((df.label == 1).sum())})")

# 2. Split + TF-IDF -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, max_df=0.9)
Xtr, Xte = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

# 3. Train 4 models -------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}
metrics = {}
for name, model in models.items():
    print(f"Training {name}...", flush=True)
    start = time.time()
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    metrics[name] = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred, average="weighted") * 100, 2),
        "recall": round(recall_score(y_test, pred, average="weighted") * 100, 2),
        "f1": round(f1_score(y_test, pred, average="weighted") * 100, 2),
        "train_seconds": round(time.time() - start, 1),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print("  ->", {k: v for k, v in metrics[name].items() if k != "confusion_matrix"})
    joblib.dump(model, MODELS / f"{name.replace(' ', '_')}.joblib", compress=3)

joblib.dump(vectorizer, MODELS / "vectorizer.joblib")
info = {"total": len(df), "fake": int((df.label == 0).sum()), "real": int((df.label == 1).sum()),
        "train_size": len(X_train), "test_size": len(X_test)}
json.dump({"models": metrics, "dataset": info}, open(MODELS / "metrics.json", "w"), indent=2)
print("Done! Models saved in the 'models' folder.")
