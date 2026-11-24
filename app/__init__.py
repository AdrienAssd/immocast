import os
from functools import partial

from flask import Flask

from app.logging_setup import configure_logging, register_request_logging
from app.model_loader import load_configured_model
from app.routes import bp


def create_app(model_path=None):
    configure_logging()
    app = Flask(__name__)
    app.config["RELOAD_TOKEN"] = os.environ.get("RELOAD_TOKEN", "")
    app.extensions["model_loader"] = partial(load_configured_model, model_path)
    register_request_logging(app)
    app.config["MODEL"] = app.extensions["model_loader"]()
    app.register_blueprint(bp)
    return app
