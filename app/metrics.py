import time

from flask import Flask, Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PRICE_BUCKETS = (4000, 6000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 16000, 18000, 20000, 25000, 30000)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests", "HTTP requests handled.", ["method", "path", "status"], registry=self.registry
        )
        self.duration = Histogram(
            "http_request_duration_seconds", "HTTP request duration.", ["path"], registry=self.registry
        )
        self.predictions = Counter(
            "predictions", "Predictions served.", ["model_version"], registry=self.registry
        )
        self.predicted_price = Histogram(
            "predicted_price_m2", "Predicted price per m² in euros.", ["model_version"],
            buckets=PRICE_BUCKETS, registry=self.registry,
        )
        self.model_info = Gauge(
            "model_info", "Model version currently served (always 1).", ["model_version"], registry=self.registry
        )
        self.feature_drift = Gauge(
            "feature_drift",
            "Total variation distance between recent inputs and the champion's training data.",
            ["feature"], registry=self.registry,
        )

    def set_model(self, version: str | None):
        self.model_info.clear()
        if version:
            self.model_info.labels(model_version=version).set(1)

    def observe_prediction(self, result: dict):
        version = str(result["model_version"])
        self.predictions.labels(model_version=version).inc()
        self.predicted_price.labels(model_version=version).observe(result["prix_m2"])


def _route_label() -> str:
    return request.url_rule.rule if request.url_rule else "unmatched"


def register_metrics(app: Flask) -> Metrics:
    metrics = Metrics()
    app.extensions["metrics"] = metrics

    @app.after_request
    def record_request(response):
        path = _route_label()
        elapsed = time.perf_counter() - g.get("request_start", time.perf_counter())
        metrics.requests.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        metrics.duration.labels(path=path).observe(elapsed)
        return response

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), mimetype=CONTENT_TYPE_LATEST)

    return metrics
