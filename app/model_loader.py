from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
import structlog
from mlflow.tracking import MlflowClient

log = structlog.get_logger("immocast.model")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["surface", "pieces", "arrondissement"]
TARGET = "prix_m2"
CHAMPION_ALIAS = "champion"
PROFILE_FILE = "training-profile.json"


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    source: str
    loaded_at: str
    run_id: str | None = None
    profile: dict | None = None


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
        log.exception("model_load_failed", source=str(path))
        return None

    log.info("model_loaded", model_version=version, source=str(path))
    return LoadedModel(model=model, version=version, source=str(path), loaded_at=_now())


def load_champion(model_name: str, alias: str = CHAMPION_ALIAS) -> LoadedModel | None:
    try:
        model_version = MlflowClient().get_model_version_by_alias(model_name, alias)
        version = str(model_version.version)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    except Exception:
        log.exception("model_load_failed", source=f"models:/{model_name}@{alias}")
        return None

    log.info("model_loaded", model_version=version, source=f"models:/{model_name}@{alias}")
    return LoadedModel(
        model=model,
        version=version,
        source=f"models:/{model_name}@{alias}",
        loaded_at=_now(),
        run_id=model_version.run_id,
        profile=load_profile(model_version.tags.get("approval_run_id"), version),
    )


def load_profile(run_id: str | None, version: str) -> dict | None:
    if not run_id:
        log.warning("training_profile_missing", model_version=version)
        return None
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_FILE}")
    except Exception:
        log.exception("training_profile_load_failed", model_version=version)
        return None


def load_configured_model(path: str | Path | None = None) -> LoadedModel | None:
    if path or os.environ.get("MODEL_PATH"):
        return load_model(path)
    model_name = os.environ.get("MODEL_NAME")
    if model_name:
        return load_champion(model_name)
    return load_model()


def predict(loaded: LoadedModel, features: dict) -> dict:
    start = time.perf_counter()
    row = pd.DataFrame([features], columns=FEATURES)
    prix_m2 = float(loaded.model.predict(row)[0])
    log.info(
        "prediction_made",
        model_version=loaded.version,
        prix_m2=round(prix_m2),
        arrondissement=features["arrondissement"],
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        n_features=len(FEATURES),
        n_missing=int(row.isna().sum().sum()),
    )
    return {
        "prix_m2": round(prix_m2),
        "prix_total": round(prix_m2 * features["surface"]),
        "model_version": loaded.version,
    }
