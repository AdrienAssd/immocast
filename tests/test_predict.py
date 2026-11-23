import pytest

from tests.conftest import PRICE_RANGE, VALID_INPUT, as_form

VALID_INPUTS = [
    pytest.param(VALID_INPUT, id="typical-apartment"),
    pytest.param({**VALID_INPUT, "surface": 9, "pieces": 1}, id="lower-bounds"),
    pytest.param({**VALID_INPUT, "surface": 400, "pieces": 10}, id="upper-bounds"),
    pytest.param({**VALID_INPUT, "code_postal": "75001"}, id="first-arrondissement"),
    pytest.param({**VALID_INPUT, "code_postal": "75020"}, id="last-arrondissement"),
    pytest.param({**VALID_INPUT, "code_postal": "75116"}, id="75116-north-of-16th"),
    pytest.param(
        {"surface": "45.5", "pieces": "2", "code_postal": " 75011 "}, id="values-as-text"
    ),
    pytest.param({**VALID_INPUT, "surface": "45,5"}, id="decimal-comma"),
    pytest.param({**VALID_INPUT, "pieces": 2.0}, id="rooms-as-whole-float"),
    pytest.param({**VALID_INPUT, "code_postal": 75011}, id="postcode-as-number"),
    pytest.param({**VALID_INPUT, "etage": 3}, id="unknown-field-ignored"),
]


@pytest.mark.parametrize("payload", VALID_INPUTS)
def test_api_returns_prices_and_model_version(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"prix_m2", "prix_total", "model_version"}
    assert isinstance(body["prix_m2"], int) and isinstance(body["prix_total"], int)
    assert PRICE_RANGE[0] <= body["prix_m2"] <= PRICE_RANGE[1]
    surface = float(str(payload["surface"]).replace(",", "."))
    assert abs(body["prix_total"] - body["prix_m2"] * surface) <= 0.5 * surface + 1
    assert body["model_version"] == client.get("/health").get_json()["model_version"]


def test_same_input_always_gives_the_same_answer(client):
    first = client.post("/predict", json=VALID_INPUT).get_json()
    second = client.post("/predict", json=VALID_INPUT).get_json()
    assert first == second


def test_75116_and_75016_are_both_the_16th_arrondissement(client):
    north = client.post("/predict", json={**VALID_INPUT, "code_postal": "75116"}).get_json()
    south = client.post("/predict", json={**VALID_INPUT, "code_postal": "75016"}).get_json()
    assert north == south


def test_web_form_shows_the_estimate(client):
    response = client.post("/", data=as_form(VALID_INPUT))

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="result"' in html and "per m²" in html
    assert 'id="error"' not in html


def test_home_page_shows_the_form(client):
    response = client.get("/")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    for field in ("surface", "pieces", "code_postal"):
        assert f'name="{field}"' in html
