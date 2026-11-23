import logging
import os

from flask import Flask

from app.model_loader import load_model
from app.routes import bp


def create_app(model_path=None):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = Flask(__name__)
    app.config["MODEL"] = load_model(model_path)
    app.register_blueprint(bp)
    return app
