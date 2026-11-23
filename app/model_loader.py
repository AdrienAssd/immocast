from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["surface", "pieces", "arrondissement"]
TARGET = "prix_m2"


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    source: str
    loaded_at: str


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)


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
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load the model from %s: %s", path, exc)
        return None

    logger.info("Loaded model %s from %s", version, path)
    return LoadedModel(
        model=model,
        version=version,
        source=str(path),
        loaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def predict(loaded: LoadedModel, features: dict) -> dict:
    prix_m2 = float(loaded.model.predict(pd.DataFrame([features], columns=FEATURES))[0])
    return {
        "prix_m2": round(prix_m2),
        "prix_total": round(prix_m2 * features["surface"]),
        "model_version": loaded.version,
    }
