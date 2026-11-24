import json

import pytest

from app import create_app
from tests.conftest import VALID_INPUT


def json_lines(output):
    lines = []
    for line in output.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines


@pytest.fixture
def logged_client(model_file, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RELOAD_TOKEN", "s3cret-token")
    app = create_app(model_file)
    app.config.update(TESTING=True)
    return app.test_client()


def test_every_response_carries_a_request_id(logged_client):
    generated = logged_client.get("/health").headers["X-Request-ID"]
    assert generated

    echoed = logged_client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"

    unsafe = logged_client.get("/health", headers={"X-Request-ID": "bad id; DROP TABLE"})
    assert unsafe.headers["X-Request-ID"] != "bad id; DROP TABLE"
    assert len(unsafe.headers["X-Request-ID"]) == 12


def test_prediction_logs_are_json_and_share_the_request_id(logged_client, capsys):
    capsys.readouterr()
    logged_client.post("/predict", json=VALID_INPUT, headers={"X-Request-ID": "req-42"})
    lines = [line for line in json_lines(capsys.readouterr().out) if line.get("request_id") == "req-42"]

    events = {line["event"]: line for line in lines}
    assert set(events) >= {"prediction_made", "request"}
    prediction = events["prediction_made"]
    assert prediction["model_version"].startswith("file-")
    assert {"prix_m2", "latency_ms", "n_features", "n_missing"} <= set(prediction)
    request = events["request"]
    assert (request["method"], request["path"], request["status"]) == ("POST", "/predict", 200)
    assert request["duration_ms"] >= 0


@pytest.mark.parametrize(
    "payload, status, level",
    [
        ({"surface": 45, "pieces": 2, "code_postal": "69001"}, 422, "warning"),
        ("not json", 400, "warning"),
    ],
)
def test_rejected_requests_are_logged_as_warnings(logged_client, capsys, payload, status, level):
    capsys.readouterr()
    if isinstance(payload, dict):
        response = logged_client.post("/predict", json=payload)
    else:
        response = logged_client.post("/predict", data=payload)
    assert response.status_code == status
    request = [line for line in json_lines(capsys.readouterr().out) if line.get("event") == "request"][-1]
    assert (request["status"], request["level"]) == (status, level)


def test_server_errors_are_logged_as_errors(client_without_model, capsys):
    capsys.readouterr()
    response = client_without_model.post("/predict", json=VALID_INPUT)
    assert response.status_code == 503
    request = [line for line in json_lines(capsys.readouterr().out) if line.get("event") == "request"][-1]
    assert request["level"] == "error"


def test_secrets_and_full_inputs_are_never_logged(logged_client, capsys):
    capsys.readouterr()
    logged_client.post("/predict", json=VALID_INPUT)
    logged_client.post("/reload", headers={"Authorization": "Bearer wrong-token"})
    logged_client.post("/reload", headers={"Authorization": "Bearer s3cret-token"})
    output = capsys.readouterr().out
    assert "s3cret-token" not in output
    assert "wrong-token" not in output
    assert "75011" not in output


def test_model_version_is_logged_at_startup(model_file, capsys):
    capsys.readouterr()
    create_app(model_file)
    loaded = [line for line in json_lines(capsys.readouterr().out) if line.get("event") == "model_loaded"]
    assert loaded and loaded[-1]["model_version"].startswith("file-")
