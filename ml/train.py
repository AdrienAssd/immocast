from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURES = ["surface", "pieces", "arrondissement"]
TARGET = "prix_m2"


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    evaluation = pd.read_csv(DATA_DIR / "evaluation.csv")

    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=42)
    model.fit(train[FEATURES], train[TARGET])

    mae = mean_absolute_error(evaluation[TARGET], model.predict(evaluation[FEATURES]))
    print(f"MAE: {mae:.0f} €/m²")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")


if __name__ == "__main__":
    main()
