import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

FEATURES = ["surface", "pieces", "arrondissement"]
TARGET = "prix_m2"

MODELS = {
    "rf": (RandomForestRegressor, {
        "n_estimators": 200,
        "min_samples_leaf": 20,
        "n_jobs": -1,
        "random_state": 42,
    }),
    "hgb": (HistGradientBoostingRegressor, {
        "loss": "absolute_error",
        "max_iter": 500,
        "learning_rate": 0.1,
        "min_samples_leaf": 100,
        "random_state": 42,
    }),
}


def model_params(name: str) -> dict:
    return dict(MODELS[name][1])


def build_model(name: str):
    cls, params = MODELS[name]
    return cls(**params)


def train(model, df: pd.DataFrame):
    model.fit(df[FEATURES], df[TARGET])
    return model


def evaluate(model, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    predictions = model.predict(df[FEATURES])
    metrics = {
        "mae": float(mean_absolute_error(df[TARGET], predictions)),
        "mape": float(mean_absolute_percentage_error(df[TARGET], predictions)),
        "n_rows": len(df),
    }
    details = df[["id_mutation", "date_mutation", *FEATURES, TARGET]].assign(prediction=predictions)
    return metrics, details
