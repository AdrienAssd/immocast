import os
from functools import partial

from flask import Flask

from app.drift_monitor import DriftMonitor
from app.logging_setup import configure_logging, register_request_logging
from app.metrics import register_metrics
from app.model_loader import load_configured_model
from app.routes import bp


def create_app(model_path=None):
    configure_logging()
    app = Flask(__name__)
    app.config["RELOAD_TOKEN"] = os.environ.get("RELOAD_TOKEN", "")
    app.extensions["model_loader"] = partial(load_configured_model, model_path)
    register_request_logging(app)
    metrics = register_metrics(app)
    app.extensions["drift"] = DriftMonitor(
        metrics.feature_drift,
        window=int(os.environ.get("DRIFT_WINDOW", "500")),
        every=int(os.environ.get("DRIFT_EVERY", "100")),
    )
    app.config["MODEL"] = app.extensions["model_loader"]()
    loaded = app.config["MODEL"]
    metrics.set_model(loaded.version if loaded else None)
    app.extensions["drift"].reset(loaded.profile if loaded else None)
    app.register_blueprint(bp)
    return app
