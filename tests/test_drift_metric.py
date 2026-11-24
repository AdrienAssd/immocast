import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient

from app import create_app
from ml.drift import build_profile
from tests.conftest import fake_model
from tests.test_drift_profile import apartments
from tests.test_metrics import metric_value

MODEL_NAME = "immocast-drift-test"
TOKEN = "drift-token"


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    uri = tmp_path_factory.mktemp("mlflow-drift").as_uri()
    with pytest.MonkeyPatch.context() as patch:
        for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_REGISTRY_URI"):
            patch.delenv(name, raising=False)
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        experiment = mlflow.create_experiment("drift-tests")

        client = MlflowClient()
        for run_name, profile in (("with-profile", build_profile(apartments())), ("without-profile", None)):
            with mlflow.start_run(experiment_id=experiment, run_name=run_name) as run:
                mlflow.sklearn.log_model(fake_model(), artifact_path="model", registered_model_name=MODEL_NAME)
                if profile:
                    mlflow.log_dict(profile, "training-profile.json")
            version = str(len(client.search_model_versions(f"name='{MODEL_NAME}'")))
            if profile:
                client.set_model_version_tag(MODEL_NAME, version, "approval_run_id", run.info.run_id)
        client.set_registered_model_alias(MODEL_NAME, "champion", "1")
        yield client


@pytest.fixture
def drift_client(registry, monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_NAME", MODEL_NAME)
    monkeypatch.setenv("RELOAD_TOKEN", TOKEN)
    monkeypatch.setenv("DRIFT_EVERY", "50")
    monkeypatch.setenv("DRIFT_WINDOW", "100")
    registry.set_registered_model_alias(MODEL_NAME, "champion", "1")
    app = create_app()
    app.config.update(TESTING=True)
    yield app.test_client()
    registry.set_registered_model_alias(MODEL_NAME, "champion", "1")


def send(client, rows):
    for row in rows.itertuples(index=False):
        payload = {"surface": float(row.surface), "pieces": int(row.pieces),
                   "code_postal": f"750{int(row.arrondissement):02d}"}
        assert client.post("/predict", json=payload).status_code == 200


def test_drift_is_published_only_after_enough_predictions(drift_client):
    send(drift_client, apartments(rows=49, seed=1))
    assert "feature_drift{" not in drift_client.get("/metrics").text

    send(drift_client, apartments(rows=1, seed=2))
    text = drift_client.get("/metrics").text
    for feature in ("surface", "pieces", "arrondissement"):
        assert f'feature_drift{{feature="{feature}"}}' in text


def test_normal_traffic_stays_below_the_threshold(drift_client):
    send(drift_client, apartments(rows=100, seed=3))
    text = drift_client.get("/metrics").text
    assert metric_value(text, "feature_drift", feature="arrondissement") < 0.20


def test_shifted_traffic_crosses_the_threshold(drift_client):
    send(drift_client, apartments(rows=100, seed=4, central=True))
    text = drift_client.get("/metrics").text
    for feature in ("surface", "pieces", "arrondissement"):
        assert metric_value(text, "feature_drift", feature=feature) >= 0.20


def test_reload_to_a_version_without_profile_clears_the_drift(drift_client, registry):
    send(drift_client, apartments(rows=50, seed=5, central=True))
    assert "feature_drift{" in drift_client.get("/metrics").text

    registry.set_registered_model_alias(MODEL_NAME, "champion", "2")
    response = drift_client.post("/reload", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.get_json()["model_version"] == "2"

    send(drift_client, apartments(rows=50, seed=6, central=True))
    assert "feature_drift{" not in drift_client.get("/metrics").text


def test_a_model_loaded_from_a_file_has_no_drift_metric(client):
    for _ in range(3):
        client.post("/predict", json={"surface": 45, "pieces": 2, "code_postal": "75011"})
    assert "feature_drift{" not in client.get("/metrics").text
