from flask import Blueprint, current_app, render_template, request

from app.model_loader import predict

bp = Blueprint("main", __name__)

PARIS_POSTAL_CODES = {f"750{a:02d}" for a in range(1, 21)} | {"75116"}


class InvalidInput(ValueError):
    pass


def parse_input(data: dict) -> dict:
    try:
        surface = float(data["surface"])
        pieces = int(data["pieces"])
    except (KeyError, TypeError, ValueError):
        raise InvalidInput("Living area and number of rooms must be numbers.")

    code_postal = str(data.get("code_postal", "")).strip()
    if code_postal not in PARIS_POSTAL_CODES:
        raise InvalidInput("Postal code must be a Paris postal code (75001 to 75020).")
    if not 9 <= surface <= 400:
        raise InvalidInput("Living area must be between 9 and 400 m².")
    if not 1 <= pieces <= 10:
        raise InvalidInput("Number of rooms must be between 1 and 10.")

    return {"surface": surface, "pieces": pieces, "arrondissement": int(code_postal[-2:])}


@bp.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    form = request.form.to_dict()
    if request.method == "POST":
        try:
            result = predict(current_app.config["MODEL"], parse_input(form))
        except InvalidInput as e:
            error = str(e)
    return render_template("index.html", result=result, error=error, form=form)
