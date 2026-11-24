import re

from tests.conftest import VALID_INPUT


def metric_value(text: str, name: str, **labels) -> float:
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        match = re.match(rf"^{re.escape(name)}(?:\{{(.*)\}})? (\S+)$", line)
        if not match:
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1) or ""))
        if all(found.get(k) == v for k, v in labels.items()):
            return float(match.group(2))
    return 0.0


def test_metrics_endpoint_lists_every_metric(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    for name in ("http_requests_total", "http_request_duration_seconds", "predictions_total",
                 "predicted_price_m2", "model_info"):
        assert f"# TYPE {name.removesuffix('_total')}" in response.text


def test_predictions_are_counted_per_model_version(client):
    version = client.get("/health").get_json()["model_version"]
    for _ in range(3):
        client.post("/predict", json=VALID_INPUT)
    client.post("/", data={key: str(value) for key, value in VALID_INPUT.items()})

    text = client.get("/metrics").text
    assert metric_value(text, "predictions_total", model_version=version) == 4
    assert metric_value(text, "predicted_price_m2_count", model_version=version) == 4
    assert metric_value(text, "model_info", model_version=version) == 1


def test_requests_are_counted_by_route_and_status(client):
    client.post("/predict", json=VALID_INPUT)
    client.post("/predict", json={"surface": 45, "pieces": 2, "code_postal": "69001"})
    client.get("/does-not-exist")

    text = client.get("/metrics").text
    assert metric_value(text, "http_requests_total", method="POST", path="/predict", status="200") == 1
    assert metric_value(text, "http_requests_total", method="POST", path="/predict", status="422") == 1
    assert metric_value(text, "http_requests_total", path="unmatched", status="404") == 1
    assert metric_value(text, "http_request_duration_seconds_count", path="/predict") == 2


def test_without_a_model_no_version_is_reported(client_without_model):
    client_without_model.post("/predict", json=VALID_INPUT)
    text = client_without_model.get("/metrics").text
    assert "model_info{" not in text
    assert metric_value(text, "http_requests_total", path="/predict", status="503") == 1
