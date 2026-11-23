from app import create_app
from app.model_loader import DEFAULT_MODEL_PATH, model_path
from tests.conftest import VALID_INPUT, as_form, make_fake_model


def test_health_is_ok_and_gives_the_version_when_a_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("file-")


def test_model_info_describes_the_loaded_model(client):
    info = client.get("/model-info").get_json()

    assert info["ready"] is True
    assert info["model_version"] == client.get("/health").get_json()["model_version"]
    assert info["model_type"] == "RandomForestRegressor"
    assert info["features"] == ["surface", "pieces", "arrondissement"]
    assert info["target"] == "prix_m2"
    assert info["model_source"] and info["loaded_at"]


def test_without_a_model_the_app_starts_but_is_not_healthy(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=VALID_INPUT).status_code == 503
    assert client_without_model.post("/", data=as_form(VALID_INPUT)).status_code == 503

    info = client_without_model.get("/model-info").get_json()
    assert info["ready"] is False and info["model_version"] is None

    assert client_without_model.get("/").status_code == 200


def test_a_corrupted_model_file_does_not_crash_the_app(tmp_path):
    broken = tmp_path / "model.joblib"
    broken.write_bytes(b"this is not a model")

    client = create_app(broken).test_client()

    assert client.get("/health").status_code == 503


def test_model_path_comes_from_the_environment(model_file, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_file))

    client = create_app().test_client()

    assert client.get("/health").status_code == 200
    assert client.get("/model-info").get_json()["model_source"] == str(model_file)


def test_a_different_model_file_gets_a_different_version(model_file, tmp_path):
    other_file = make_fake_model(tmp_path / "other.joblib", seed=1)

    same = create_app(model_file).test_client().get("/health").get_json()["model_version"]
    again = create_app(model_file).test_client().get("/health").get_json()["model_version"]
    other = create_app(other_file).test_client().get("/health").get_json()["model_version"]

    assert same == again
    assert other != same


def test_an_empty_model_path_means_the_default_file(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "")
    assert model_path() == DEFAULT_MODEL_PATH
