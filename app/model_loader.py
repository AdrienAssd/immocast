from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["surface", "pieces", "arrondissement"]
TARGET = "prix_m2"
CHAMPION_ALIAS = "champion"


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    source: str
    loaded_at: str
    run_id: str | None = None


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _file_version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return f"file-{digest.hexdigest()[:12]}"


def load_model(path: str | Path | None = None) -> LoadedModel | None:
    path = Path(path) if path else model_path()
    try:
        model = joblib.load(path)
        version = _file_version(path)
    except Exception:
        logger.exception("Could not load the model from %s", path)
        return None

    logger.info("Loaded model %s from %s", version, path)
    return LoadedModel(model=model, version=version, source=str(path), loaded_at=_now())


def load_champion(model_name: str, alias: str = CHAMPION_ALIAS) -> LoadedModel | None:
    try:
        model_version = MlflowClient().get_model_version_by_alias(model_name, alias)
        version = str(model_version.version)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    except Exception:
        logger.exception("Could not load %s@%s from MLflow", model_name, alias)
        return None

    logger.info("Loaded %s version %s (alias %s)", model_name, version, alias)
    return LoadedModel(
        model=model,
        version=version,
        source=f"models:/{model_name}@{alias}",
        loaded_at=_now(),
        run_id=model_version.run_id,
    )


def load_configured_model(path: str | Path | None = None) -> LoadedModel | None:
    if path or os.environ.get("MODEL_PATH"):
        return load_model(path)
    model_name = os.environ.get("MODEL_NAME")
    if model_name:
        return load_champion(model_name)
    return load_model()


def predict(loaded: LoadedModel, features: dict) -> dict:
    prix_m2 = float(loaded.model.predict(pd.DataFrame([features], columns=FEATURES))[0])
    return {
        "prix_m2": round(prix_m2),
        "prix_total": round(prix_m2 * features["surface"]),
        "model_version": loaded.version,
    }
