import logging
import os
from functools import partial

from flask import Flask

from app.model_loader import load_configured_model
from app.routes import bp


def create_app(model_path=None):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = Flask(__name__)
    app.config["RELOAD_TOKEN"] = os.environ.get("RELOAD_TOKEN", "")
    app.extensions["model_loader"] = partial(load_configured_model, model_path)
    app.config["MODEL"] = app.extensions["model_loader"]()
    app.register_blueprint(bp)
    return app
