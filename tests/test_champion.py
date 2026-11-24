import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor

from app import create_app
from app.model_loader import FEATURES
from tests.conftest import VALID_INPUT, fake_model, make_fake_model

MODEL_NAME = "immocast-test"
TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CANDIDATE_PRICE = 1.0


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    uri = tmp_path_factory.mktemp("mlflow").as_uri()
    with pytest.MonkeyPatch.context() as patch:
        for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_REGISTRY_URI"):
            patch.delenv(name, raising=False)
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)

        experiment = mlflow.create_experiment("tests")
        sample = pd.DataFrame([[45.0, 2, 11]], columns=FEATURES)
        candidate = DummyRegressor(strategy="constant", constant=CANDIDATE_PRICE).fit(sample, [1.0])
        for run_name, model in (("champion", fake_model()), ("newer-candidate", candidate)):
            with mlflow.start_run(experiment_id=experiment, run_name=run_name):
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        client = MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "champion", "1")
        yield client


@pytest.fixture
def champion_restored(registry):
    yield registry
    registry.set_registered_model_alias(MODEL_NAME, "champion", "1")


@pytest.fixture
def registry_env(registry, monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_NAME", MODEL_NAME)
    monkeypatch.setenv("RELOAD_TOKEN", TOKEN)


def new_client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_champion_is_served_even_when_a_newer_version_exists(registry_env):
    client = new_client()

    assert client.get("/health").get_json() == {"status": "ok", "model_version": "1"}
    prediction = client.post("/predict", json=VALID_INPUT).get_json()
    assert prediction["model_version"] == "1"
    assert prediction["prix_m2"] != CANDIDATE_PRICE
    info = client.get("/model-info").get_json()
    assert info["model_source"] == f"models:/{MODEL_NAME}@champion"
    assert info["run_id"]


def test_reload_serves_the_new_champion(registry_env, champion_restored):
    client = new_client()
    champion_restored.set_registered_model_alias(MODEL_NAME, "champion", "2")

    assert client.get("/health").get_json()["model_version"] == "1"
    response = client.post("/reload", headers=AUTH)

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "reloaded", "previous_version": "1", "model_version": "2",
    }
    prediction = client.post("/predict", json=VALID_INPUT).get_json()
    assert prediction["model_version"] == "2"
    assert prediction["prix_m2"] == CANDIDATE_PRICE


def test_without_champion_the_app_waits_then_reload_bootstraps_it(registry_env, champion_restored):
    champion_restored.delete_registered_model_alias(MODEL_NAME, "champion")
    client = new_client()
    assert client.get("/health").status_code == 503

    champion_restored.set_registered_model_alias(MODEL_NAME, "champion", "1")
    response = client.post("/reload", headers=AUTH)

    assert response.status_code == 200
    assert response.get_json()["previous_version"] is None
    assert client.get("/health").get_json() == {"status": "ok", "model_version": "1"}


def test_failed_reload_keeps_serving_the_current_model(registry_env, champion_restored):
    client = new_client()
    champion_restored.delete_registered_model_alias(MODEL_NAME, "champion")

    response = client.post("/reload", headers=AUTH)

    assert response.status_code == 503
    assert response.get_json() == {"status": "failed", "model_version": "1"}
    assert client.get("/health").get_json() == {"status": "ok", "model_version": "1"}


def test_unknown_model_name_is_not_healthy(registry_env, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "model-that-does-not-exist")

    assert new_client().get("/health").status_code == 503


@pytest.mark.parametrize(
    "configured_token, authorization, expected_status",
    [
        pytest.param("", None, 404, id="disabled-without-token"),
        pytest.param("", "Bearer anything", 404, id="disabled-even-with-a-header"),
        pytest.param(TOKEN, None, 401, id="missing-header"),
        pytest.param(TOKEN, "Bearer wrong", 401, id="wrong-token"),
        pytest.param(TOKEN, TOKEN, 401, id="token-without-bearer"),
    ],
)
def test_reload_is_protected(registry_env, monkeypatch, configured_token, authorization,
                             expected_status):
    monkeypatch.setenv("RELOAD_TOKEN", configured_token)
    client = new_client()
    headers = {"Authorization": authorization} if authorization else {}

    assert client.post("/reload", headers=headers).status_code == expected_status
    assert client.get("/health").get_json()["model_version"] == "1"


def test_model_path_takes_priority_over_model_name(registry_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(make_fake_model(tmp_path / "model.joblib")))
    client = new_client()

    version = client.get("/health").get_json()["model_version"]
    assert version.startswith("file-")
    response = client.post("/reload", headers=AUTH)
    assert response.get_json() == {
        "status": "reloaded", "previous_version": version, "model_version": version,
    }
