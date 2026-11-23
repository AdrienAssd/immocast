import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from app import create_app
from app.model_loader import FEATURES

VALID_INPUT = {"surface": 45, "pieces": 2, "code_postal": "75011"}
PRICE_RANGE = (3000, 30000)


def make_fake_model(path, seed=0):
    rng = np.random.default_rng(seed)
    rows = 500
    features = pd.DataFrame(
        {
            "surface": rng.uniform(9, 400, rows),
            "pieces": rng.integers(1, 11, rows),
            "arrondissement": rng.integers(1, 21, rows),
        }
    )[FEATURES]
    prix_m2 = (
        14_000 - 250 * features["arrondissement"] - 4 * features["surface"]
        + rng.normal(0, 500, rows)
    ).clip(*PRICE_RANGE)
    model = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, random_state=seed)
    model.fit(features, prix_m2)
    joblib.dump(model, path)
    return path


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    return make_fake_model(tmp_path_factory.mktemp("model") / "model.joblib")


@pytest.fixture
def client(model_file):
    app = create_app(model_file)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def client_without_model(tmp_path):
    app = create_app(tmp_path / "missing.joblib")
    app.config.update(TESTING=True)
    return app.test_client()


def as_form(data):
    return {key: str(value) for key, value in data.items()}
