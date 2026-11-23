from flask import Flask

from app.model_loader import load_model
from app.routes import bp


def create_app():
    app = Flask(__name__)
    app.config["MODEL"] = load_model()
    app.register_blueprint(bp)
    return app
