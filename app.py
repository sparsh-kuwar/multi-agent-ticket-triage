"""Flask REST API and web UI for the multi-agent ticket triage system.

Run locally:  flask --app app run --debug
Production:   gunicorn app:app
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from triage import agents, llm  # noqa: E402  (reads env vars loaded above)
from triage.pipeline import TicketStore, triage  # noqa: E402

logging.basicConfig(level=logging.INFO)

MAX_TICKET_CHARS = 5000

SAMPLE_TICKETS = [
    ("Priya S.", "I was charged twice for my subscription this month. Please refund the extra charge."),
    ("Acme Corp", "URGENT: our whole team cannot access the dashboard, it shows error 500 since this morning. "
                  "Production is blocked!"),
    ("Rahul K.", "Where is my package? Tracking hasn't updated in 5 days and the courier says nothing."),
    ("Meera J.", "Just wondering if you offer a student discount on the annual plan. No rush, thanks!"),
    ("Dev P.", "This is the third time I'm asking. My account is still locked and nobody replies. Ridiculous!!!"),
    ("Anita R.", "I think my account was hacked, there are logins I don't recognise and an unauthorized payment."),
]


def create_app(seed: bool | None = None) -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False
    store = TicketStore()
    app.extensions["tickets"] = store

    if seed if seed is not None else os.environ.get("SEED_SAMPLE_TICKETS", "true").lower() == "true":
        for customer, text in SAMPLE_TICKETS:
            store.add(text, customer, triage(text))

    def error(status: int, message: str):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        return render_template("index.html", samples=[t for _, t in SAMPLE_TICKETS])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "llm" if llm.enabled() else "rules",
                "model": llm.model_name() if llm.enabled() else None}

    @app.post("/api/triage")
    def triage_only():
        """Classify a ticket without storing it (for integrations that keep their own records)."""
        text, err = _ticket_text()
        if err:
            return error(*err)
        return triage(text)

    @app.post("/api/tickets")
    def create_ticket():
        text, err = _ticket_text()
        if err:
            return error(*err)
        customer = (request.get_json(silent=True) or {}).get("customer")
        customer = str(customer).strip()[:100] if customer else None
        record = store.add(text, customer, triage(text))
        return record, 201

    @app.get("/api/tickets")
    def list_tickets():
        escalated = request.args.get("escalated")
        if escalated not in (None, "true", "false"):
            return error(400, "escalated must be true or false")
        team = request.args.get("team")
        if team and team not in agents.TEAMS.values():
            return error(400, f"team must be one of: {', '.join(sorted(set(agents.TEAMS.values())))}")
        items = store.list(None if escalated is None else escalated == "true", team)
        return {"count": len(items), "items": items}

    @app.get("/api/tickets/<int:tid>")
    def get_ticket(tid: int):
        record = store.get(tid)
        return record if record else error(404, f"ticket {tid} not found")

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.errorhandler(404)
    def not_found(_):
        return error(404, "not found")

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error(405, "method not allowed")

    return app


def _ticket_text():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (400, 'send JSON like {"ticket": "..."}')
    text = data.get("ticket")
    if not isinstance(text, str) or not text.strip():
        return None, (422, "ticket must be a non-empty string")
    if len(text) > MAX_TICKET_CHARS:
        return None, (422, f"ticket must be at most {MAX_TICKET_CHARS} characters")
    return text.strip(), None


app = create_app()
