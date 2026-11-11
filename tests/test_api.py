"""The service, tested end to end with the real model file and embedding model."""

import pytest
from fastapi.testclient import TestClient

from handover.api import MAX_CHARS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Handover" in r.text


def test_info_lists_teams_and_settings(client):
    info = client.get("/info").json()
    assert len(info["teams"]) == 10
    assert "Billing and Payments" in info["teams"]
    assert info["operating_points"]


def test_route_returns_a_consistent_decision(client):
    r = client.post("/route", json={"text": "Meine Rechnung wurde doppelt abgebucht, bitte erstatten."})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"route", "handover"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert sum(v["neighbours"] for v in body["votes"]) == 5
    assert body["precedent"]["team"] == body["votes"][0]["team"] or len(body["votes"]) > 1
    assert 0.0 < body["precedent"]["similarity"] <= 1.001
    # The team is given only when the system takes the decision itself.
    assert (body["team"] is None) == (body["decision"] == "handover")
    if body["decision"] == "route":
        assert body["confidence"] >= body["threshold"]
        assert body["reason"] == "similar past tickets agree"


def test_stricter_setting_never_routes_what_a_looser_one_hands_over(client):
    info = client.get("/info").json()
    targets = sorted(p["target_risk"] for p in info["operating_points"] if p["threshold"] is not None)
    if len(targets) < 2:
        pytest.skip("only one usable setting")
    text = "The VPN keeps disconnecting on my laptop since this morning."
    strict = client.post("/route", json={"text": text, "target_risk": targets[0]}).json()
    loose = client.post("/route", json={"text": text, "target_risk": targets[-1]}).json()
    assert not (strict["decision"] == "route" and loose["decision"] == "handover")


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * (MAX_CHARS + 1)}])
def test_bad_input_is_rejected(client, payload):
    assert client.post("/route", json=payload).status_code == 422


def test_unknown_setting_is_rejected(client):
    assert client.post("/route", json={"text": "hello there", "target_risk": 0.123}).status_code == 422


def test_a_ticket_already_in_memory_is_routed_like_its_copy(client):
    """A message identical to a past ticket must find it and reuse its team."""
    from handover.model import Router

    router = Router.load()
    text, team = router.index_texts[0], router.classes[router.index_labels[0]]
    body = client.post("/route", json={"text": text}).json()
    assert body["precedent"]["similarity"] > 0.999
    assert body["precedent"]["team"] == team


@pytest.mark.parametrize(
    "text",
    [
        "asdf qwer zxcv 12345 lorem ipsum",
        "Hallo, es funktioniert nicht. Können Sie mir bitte helfen?",
        "Hi, I have a question about my account. Thanks.",
        "Wann ist das Wetter schön in Dortmund?",
    ],
)
def test_messages_unlike_anything_seen_are_handed_over(client, text):
    """Even at the most lenient setting, an unfamiliar message goes to a person."""
    info = client.get("/info").json()
    loosest = max(p["target_risk"] for p in info["operating_points"] if p["threshold"] is not None)
    body = client.post("/route", json={"text": text, "target_risk": loosest}).json()
    assert body["decision"] == "handover"
    assert body["reason"] == "no similar past ticket"
    assert body["team"] is None
