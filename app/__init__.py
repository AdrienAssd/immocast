"""ImmoCast web app.

Configuration comes from environment variables:
    MODEL_PATH   model file saved by ml/train.py (default: models/model.joblib)
    LOG_LEVEL    default: INFO
"""

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
    # None if the model cannot be loaded: the app still starts and /health answers 503.
    app.config["MODEL"] = load_model(model_path)
    app.register_blueprint(bp)
    return app
