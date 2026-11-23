"""Load the model and run predictions.

For now the model is the scikit-learn estimator saved by ml/train.py with joblib.
A later PR replaces the file by the MLflow registry (alias champion); the rest of the app
only uses LoadedModel and predict(), so it will not change.
"""

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
    version: str  # identifies exactly which model answers
    source: str  # where it was loaded from
    loaded_at: str


def model_path() -> Path:
    """MODEL_PATH environment variable, or models/model.joblib in the repository."""
    return Path(os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)  # empty = not set


def _file_version(path: Path) -> str:
    """Short SHA-256 of the file: two different model files never get the same version."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return f"file-{digest.hexdigest()[:12]}"


def load_model(path: str | Path | None = None) -> LoadedModel | None:
    """Load the model, or return None if it cannot be loaded (the app then answers 503)."""
    path = Path(path) if path else model_path()
    try:
        model = joblib.load(path)
        version = _file_version(path)
    except Exception as exc:  # noqa: BLE001 - a corrupted file can raise almost anything
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
