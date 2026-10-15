import pytest

from triage import agents, llm, rules
from triage.pipeline import TicketStore, triage


@pytest.fixture(autouse=True)
def rules_only(monkeypatch):
    """Tests never call a real LLM."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRIAGE_USE_LLM", "false")


@pytest.fixture
def client():
    from app import create_app

    app = create_app(seed=False)
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------- rule classifiers

@pytest.mark.parametrize("text, intent", [
    ("I was charged twice on my invoice", "Billing"),
    ("I want my money back for this order, please refund", "Refund"),
    ("The app crashes with error 500 when I open it", "Technical Issue"),
    ("My account is locked and the password reset fails to log in", "Account"),
    ("My package was not delivered and tracking is stuck", "Shipping"),
    ("Hello there", "General"),
])
def test_intent(text, intent):
    assert rules.classify_intent(text)["label"] == intent


@pytest.mark.parametrize("text, priority", [
    ("My account was hacked and there is an unauthorized payment", "URGENT"),
    ("URGENT: production is down for all users", "URGENT"),
    ("I was double charged this month", "HIGH"),
    ("The export button is not working", "MEDIUM"),
    ("Just wondering about pricing, no rush", "LOW"),
])
def test_priority(text, priority):
    assert rules.classify_priority(text)["label"] == priority


@pytest.mark.parametrize("text, sentiment", [
    ("Thanks, I really appreciate the help!", "Positive"),
    ("Where can I change my email address?", "Neutral"),
    ("This is the third time I'm asking and still no response", "Frustrated"),
    ("This is ridiculous, worst service ever!!!", "Angry"),
    ("THIS IS COMPLETELY UNACCEPTABLE AND I WANT IT FIXED", "Angry"),
    ("Not great, the update broke my settings again", "Frustrated"),
])
def test_sentiment(text, sentiment):
    assert rules.classify_sentiment(text)["label"] == sentiment


def test_no_signal_gives_low_intent_confidence_and_default_priority():
    assert rules.classify_intent("hello")["confidence"] < agents.CONFIDENCE_FLOOR
    priority = rules.classify_priority("hello")
    assert priority["label"] == "MEDIUM"
    assert priority["confidence"] >= agents.CONFIDENCE_FLOOR


def test_short_keywords_match_whole_words_only():
    # "bill" must not match inside "billion"; "app" must not match inside "happy".
    assert "bill" not in rules.classify_intent("a billion stars")["evidence"]
    assert "app" not in rules.classify_intent("I am happy")["evidence"]


# ---------------------------------------------------------------- routing and escalation

def test_routine_ticket_is_auto_responded():
    r = triage("My package was not delivered yet, the tracking number shows the courier has it")
    assert r["routing"]["team"] == "Shipping Support"
    assert r["routing"]["escalate"] is False
    assert r["routing"]["auto_response_sent"] is True
    assert r["mode"] == "rules"


def test_urgent_ticket_is_escalated_with_reason():
    r = triage("My account was hacked and there is an unauthorized payment")
    assert r["routing"]["escalate"] is True
    assert "URGENT priority" in r["routing"]["escalation_reasons"]
    assert r["routing"]["sla_hours"] == 1
    assert "senior" in r["routing"]["first_response"]


def test_angry_customer_is_escalated_and_gets_apology():
    r = triage("Worst service ever, my refund is ridiculous")
    assert r["routing"]["escalate"] is True
    assert "angry customer" in r["routing"]["escalation_reasons"]
    assert r["routing"]["first_response"].startswith("Sorry")


def test_uncertain_ticket_is_escalated():
    r = triage("hello")
    assert r["routing"]["escalate"] is True
    assert any("low" in reason for reason in r["routing"]["escalation_reasons"])


def test_every_intent_maps_to_a_team():
    assert set(agents.TEAMS) == set(rules.INTENTS)


# ---------------------------------------------------------------- LLM fallback

def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRIAGE_USE_LLM", "true")
    monkeypatch.setattr(llm, "_client", lambda: (_ for _ in ()).throw(RuntimeError("network down")))
    r = triage("The app crashes with error 500")
    assert r["intent"]["source"] == "rules"
    assert r["intent"]["label"] == "Technical Issue"


def test_llm_answer_is_used_when_valid(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda prompt: {"label": "refund", "confidence": 0.93, "evidence": ["money back"]})
    result = agents.intent_agent("anything")
    assert result == {"label": "Refund", "confidence": 0.93, "evidence": ["money back"], "source": "llm"}


@pytest.mark.parametrize("bad", [{"label": "Pizza", "confidence": 0.9}, {"label": "Refund", "confidence": "high"}, None])
def test_invalid_llm_answer_falls_back(monkeypatch, bad):
    monkeypatch.setattr(llm, "ask_json", lambda prompt: bad)
    assert agents.intent_agent("please refund my order")["source"] == "rules"


def test_llm_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRIAGE_USE_LLM", "false")
    assert llm.enabled() is False


# ---------------------------------------------------------------- store

def test_store_is_bounded_and_newest_first():
    store = TicketStore(limit=2)
    for i in range(3):
        store.add(f"ticket {i}", None, triage("Where is my package?"))
    items = store.list()
    assert [t["ticket"] for t in items] == ["ticket 2", "ticket 1"]


# ---------------------------------------------------------------- HTTP API

def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok", "mode": "rules", "model": None}


def test_create_get_list_and_stats(client):
    resp = client.post("/api/tickets", json={"ticket": "I was charged twice", "customer": " Priya "})
    assert resp.status_code == 201
    created = resp.get_json()
    assert created["customer"] == "Priya"
    assert created["routing"]["team"] == "Payments Support"

    assert client.get(f"/api/tickets/{created['id']}").get_json()["id"] == created["id"]
    client.post("/api/tickets", json={"ticket": "Where is my parcel? Tracking says delivered"})
    assert client.get("/api/tickets").get_json()["count"] == 2
    escalated = client.get("/api/tickets?escalated=true").get_json()
    assert all(t["routing"]["escalate"] for t in escalated["items"])
    assert client.get("/api/stats").get_json()["total"] == 2


def test_triage_endpoint_does_not_store(client):
    resp = client.post("/api/triage", json={"ticket": "My app crashes"})
    assert resp.status_code == 200
    assert "routing" in resp.get_json()
    assert client.get("/api/tickets").get_json()["count"] == 0


@pytest.mark.parametrize("body, status", [
    (None, 400), ({"ticket": ""}, 422), ({"ticket": 5}, 422), ({"ticket": "x" * 5001}, 422),
])
def test_bad_input(client, body, status):
    resp = client.post("/api/tickets", json=body) if body is not None else client.post("/api/tickets", data="nope")
    assert resp.status_code == status
    assert "error" in resp.get_json()


def test_bad_filters_and_missing_ticket(client):
    assert client.get("/api/tickets?escalated=maybe").status_code == 400
    assert client.get("/api/tickets?team=Nope").status_code == 400
    assert client.get("/api/tickets/999999").status_code == 404
    assert client.get("/api/nothing").get_json() == {"error": "not found"}


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Ticket queue" in resp.data


def test_seeded_app_has_samples():
    from app import SAMPLE_TICKETS, create_app

    app = create_app(seed=True)
    stats = app.test_client().get("/api/stats").get_json()
    assert stats["total"] == len(SAMPLE_TICKETS)
    assert stats["escalated"] >= 2
    assert stats["auto_resolved"] >= 2
