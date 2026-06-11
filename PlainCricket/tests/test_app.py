"""Route-level tests that don't touch the network."""

import pytest

from index import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_info(client):
    res = client.get("/api")
    assert res.status_code == 200
    data = res.get_json()
    assert data["app"] == "PlainCricket API"
    assert "/api/matches" in data["endpoints"]


def test_score_requires_id(client):
    res = client.get("/api/score")
    assert res.status_code == 400
    assert "numeric" in res.get_json()["error"]


def test_score_rejects_non_numeric_id(client):
    res = client.get("/api/score?id=12345/../evil")
    assert res.status_code == 400


def test_error_responses_are_not_cached(client):
    res = client.get("/api/score")
    assert res.headers["Cache-Control"] == "no-store"


def test_unknown_endpoint_is_json_404(client):
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.get_json()["error"] == "Endpoint not found"


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"PlainCricket" in res.data
