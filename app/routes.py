import hmac
import math
import os
import threading

import structlog
from flask import Blueprint, current_app, jsonify, render_template, request

from app.model_loader import FEATURES, TARGET, predict

bp = Blueprint("main", __name__)
log = structlog.get_logger("immocast")
_reload_lock = threading.Lock()

PARIS_POSTAL_CODES = {f"750{a:02d}" for a in range(1, 21)} | {"75116"}
SURFACE_MIN, SURFACE_MAX = 9, 400
PIECES_MIN, PIECES_MAX = 1, 10


class InvalidInput(ValueError):

    def __init__(self, errors: dict):
        super().__init__(" ".join(errors.values()))
        self.errors = errors


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_input(data: dict) -> dict:
    errors = {}

    surface = _number(data.get("surface"))
    if _is_missing(data.get("surface")):
        errors["surface"] = "Living area is required."
    elif surface is None:
        errors["surface"] = "Living area must be a number."
    elif not SURFACE_MIN <= surface <= SURFACE_MAX:
        errors["surface"] = f"Living area must be between {SURFACE_MIN} and {SURFACE_MAX} m²."

    pieces = _number(data.get("pieces"))
    if _is_missing(data.get("pieces")):
        errors["pieces"] = "Number of rooms is required."
    elif pieces is None or not pieces.is_integer():
        errors["pieces"] = "Number of rooms must be a whole number."
    elif not PIECES_MIN <= pieces <= PIECES_MAX:
        errors["pieces"] = f"Number of rooms must be between {PIECES_MIN} and {PIECES_MAX}."

    code_postal = data.get("code_postal")
    if isinstance(code_postal, int) and not isinstance(code_postal, bool):
        code_postal = str(code_postal)  # 75011 sent as a number: Paris codes have no leading zero
    code_postal = code_postal.strip() if isinstance(code_postal, str) else ""
    if code_postal not in PARIS_POSTAL_CODES:
        errors["code_postal"] = (
            "Postal code must be a Paris postal code (75001 to 75020, or 75116)."
        )

    if errors:
        raise InvalidInput(errors)
    return {"surface": surface, "pieces": int(pieces), "arrondissement": int(code_postal[-2:])}


def _model():
    return current_app.config.get("MODEL")


@bp.route("/", methods=["GET", "POST"])
def index():
    result, error, status = None, None, 200
    form = request.form.to_dict()
    loaded = _model()
    if request.method == "POST":
        if loaded is None:
            error = "The model is not available right now. Please try again later."
            status = 503
        else:
            try:
                result = predict(loaded, parse_input(form))
            except InvalidInput as e:
                error = str(e)
    page = render_template("index.html", result=result, error=error, form=form, model=loaded)
    return page, status


@bp.post("/predict")
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        example = '{"surface": 45, "pieces": 2, "code_postal": "75011"}'
        return jsonify(error=f"Send a JSON object, for example {example}."), 400

    loaded = _model()
    if loaded is None:
        return jsonify(error="The model is not loaded."), 503

    try:
        features = parse_input(data)
    except InvalidInput as e:
        return jsonify(error=str(e), errors=e.errors), 422
    return jsonify(predict(loaded, features))


@bp.get("/health")
def health():
    """200 only when a model is loaded and can answer; 503 otherwise."""
    loaded = _model()
    if loaded is None:
        return jsonify(status="unavailable", reason="model not loaded"), 503
    return jsonify(status="ok", model_version=loaded.version)


@bp.get("/model-info")
def model_info():
    loaded = _model()
    return jsonify(
        ready=loaded is not None,
        model_version=loaded.version if loaded else None,
        model_source=loaded.source if loaded else None,
        model_type=type(loaded.model).__name__ if loaded else None,
        loaded_at=loaded.loaded_at if loaded else None,
        run_id=loaded.run_id if loaded else None,
        features=FEATURES,
        target=TARGET,
        git_sha=os.environ.get("GIT_SHA"),
    )


@bp.post("/reload")
def reload_model():
    token = current_app.config.get("RELOAD_TOKEN")
    if not token:
        return jsonify(error="Reload is disabled."), 404
    provided = request.headers.get("Authorization", "")
    if not hmac.compare_digest(provided.encode(), f"Bearer {token}".encode()):
        log.warning("reload_refused")
        return jsonify(error="Invalid or missing token."), 401

    with _reload_lock:
        previous = _model()
        previous_version = previous.version if previous else None
        loaded = current_app.extensions["model_loader"]()
        if loaded is None:
            log.error("model_reload_failed", model_version=previous_version)
            return jsonify(status="failed", model_version=previous_version), 503
        current_app.config["MODEL"] = loaded
    log.info("model_reloaded", previous_version=previous_version, model_version=loaded.version)
    return jsonify(
        status="reloaded", previous_version=previous_version, model_version=loaded.version
    )
