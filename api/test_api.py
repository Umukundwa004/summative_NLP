"""Tests for the API.  Run them from the project folder:

    pip install pytest httpx
    pytest api -q

They use the models that the notebook wrote into the models/ folder (they are skipped if there is none yet).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import MODELS_DIR, app  # noqa: E402

pytestmark = pytest.mark.skipif(not list(MODELS_DIR.glob("*/config.json")), reason="no trained model yet (run the notebook)")

SAMPLE = "Ikipe y'igihugu yatsinze umukino ejo mu kibuga."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # starting the client loads the models
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["models_loaded"]) >= 1


def test_languages(client):
    languages = client.get("/languages").json()
    assert len(languages) >= 1
    for language in languages:
        assert language["dataset"] in ("kinnews", "kirnews")
        assert len(language["labels"]) >= 2
        assert 0 <= language["test_macro_f1"] <= 1


def test_predict_without_language_uses_the_first_model(client):
    response = client.post("/predict", json={"text": SAMPLE})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in body["probabilities"]
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    # probabilities are sorted, highest first
    values = list(body["probabilities"].values())
    assert values == sorted(values, reverse=True)


def test_predict_every_language_by_key_and_by_name(client):
    for language in client.get("/languages").json():
        for name in (language["dataset"], language["language"], language["language"].upper()):
            body = client.post("/predict", json={"text": SAMPLE, "language": name}).json()
            assert body["dataset"] == language["dataset"]
            assert body["label"] in language["labels"]


def test_unknown_language_gives_404_and_lists_the_available_ones(client):
    response = client.post("/predict", json={"text": SAMPLE, "language": "swahili"})
    assert response.status_code == 404
    assert "Available" in response.json()["detail"]


def test_missing_or_empty_text_gives_422(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_text_that_is_empty_after_cleaning_gives_422(client):
    for text in ["   ", "https://example.com/page", "\u200b\u200b"]:
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 422, text
        assert "empty" in response.json()["detail"].lower()


def test_text_that_is_too_long_gives_422(client):
    assert client.post("/predict", json={"text": "a " * 10001}).status_code == 422


def test_a_long_article_still_works(client):
    response = client.post("/predict", json={"text": (SAMPLE + " ") * 400})
    assert response.status_code == 200


def test_unicode_and_odd_characters_do_not_crash(client):
    for text in ["Ikipe \u2019ya \u00e9", "\U0001F600 \U0001F600", "ሰላም ዓለም", "1234 5678", "<b>html</b> &amp; text"]:
        assert client.post("/predict", json={"text": text}).status_code in (200, 422)


def test_cleaned_text_is_returned(client):
    body = client.post("/predict", json={"text": "HELLO   &amp;  World https://x.org/a"}).json()
    assert body["cleaned_text"] == "hello & world"


def test_cors_allows_a_browser_app(client):
    response = client.options("/predict", headers={"Origin": "http://localhost:5000",
                                                   "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "content-type"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", "http://localhost:5000")


def test_the_web_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Classify" in response.text
