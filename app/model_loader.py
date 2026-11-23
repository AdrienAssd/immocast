from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["surface", "pieces", "arrondissement"]


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, features: dict) -> dict:
    prix_m2 = float(model.predict(pd.DataFrame([features], columns=FEATURES))[0])
    return {"prix_m2": round(prix_m2), "prix_total": round(prix_m2 * features["surface"])}
