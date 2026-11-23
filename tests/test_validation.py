import pytest

from tests.conftest import VALID_INPUT, as_form


def without(field):
    return {key: value for key, value in VALID_INPUT.items() if key != field}


INVALID_INPUTS = [
    pytest.param("surface", without("surface"), id="surface-missing"),
    pytest.param("surface", {**VALID_INPUT, "surface": ""}, id="surface-empty"),
    pytest.param("surface", {**VALID_INPUT, "surface": None}, id="surface-null"),
    pytest.param("surface", {**VALID_INPUT, "surface": "abc"}, id="surface-text"),
    pytest.param("surface", {**VALID_INPUT, "surface": "nan"}, id="surface-nan"),
    pytest.param("surface", {**VALID_INPUT, "surface": "inf"}, id="surface-infinite"),
    pytest.param("surface", {**VALID_INPUT, "surface": True}, id="surface-boolean"),
    pytest.param("surface", {**VALID_INPUT, "surface": -10}, id="surface-negative"),
    pytest.param("surface", {**VALID_INPUT, "surface": 8.9}, id="surface-below-9"),
    pytest.param("surface", {**VALID_INPUT, "surface": 400.5}, id="surface-above-400"),
    pytest.param("pieces", without("pieces"), id="rooms-missing"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": 0}, id="rooms-zero"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": 11}, id="rooms-above-10"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": 2.5}, id="rooms-not-whole"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": "2.5"}, id="rooms-not-whole-text"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": True}, id="rooms-boolean"),
    pytest.param("pieces", {**VALID_INPUT, "pieces": "two"}, id="rooms-word"),
    pytest.param("code_postal", without("code_postal"), id="postcode-missing"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": "75021"}, id="postcode-75021"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": "75000"}, id="postcode-75000"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": "69001"}, id="postcode-not-paris"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": "7501"}, id="postcode-too-short"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": "750111"}, id="postcode-too-long"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": 75011.0}, id="postcode-float"),
    pytest.param("code_postal", {**VALID_INPUT, "code_postal": True}, id="postcode-boolean"),
]


@pytest.mark.parametrize("field, payload", INVALID_INPUTS)
def test_api_reports_the_faulty_field(client, field, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.get_json()
    assert set(body["errors"]) == {field}
    assert body["error"]  # readable summary


def test_api_reports_every_faulty_field_at_once(client):
    response = client.post("/predict", json={"surface": 2, "pieces": 0, "code_postal": "69001"})

    assert response.status_code == 422
    assert set(response.get_json()["errors"]) == {"surface", "pieces", "code_postal"}


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"data": "{not json", "content_type": "application/json"}, id="broken-json"),
        pytest.param({"json": [VALID_INPUT]}, id="json-list"),
        pytest.param({"json": "75011"}, id="json-string"),
        pytest.param({"data": ""}, id="empty-body"),
        pytest.param({"data": VALID_INPUT}, id="form-encoded-not-json"),
    ],
)
def test_api_rejects_a_body_that_is_not_a_json_object(client, request_kwargs):
    response = client.post("/predict", **request_kwargs)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_web_form_shows_the_error_and_keeps_what_was_typed(client):
    response = client.post("/", data=as_form({**VALID_INPUT, "surface": "5"}))

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="error"' in html and "between 9 and 400" in html
    assert 'value="75011"' in html
    assert 'id="result"' not in html
