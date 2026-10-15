"""Runs the agents in order and keeps an in-memory log of triaged tickets."""

import itertools
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

from . import agents, llm

MAX_TICKETS = 500


def triage(ticket: str) -> dict:
    """Classify one ticket and decide routing. Pure function: no storage."""
    started = time.perf_counter()
    intent = agents.intent_agent(ticket)
    priority = agents.priority_agent(ticket)
    sentiment = agents.sentiment_agent(ticket)
    routing = agents.routing_agent(ticket, intent, priority, sentiment)
    return {
        "intent": intent,
        "priority": priority,
        "sentiment": sentiment,
        "routing": routing,
        "mode": "llm" if llm.enabled() else "rules",
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


class TicketStore:
    """Thread-safe, bounded, in-memory store. Data is lost when the process restarts."""

    def __init__(self, limit: int = MAX_TICKETS):
        self._items: "OrderedDict[int, dict]" = OrderedDict()
        self._ids = itertools.count(1001)
        self._lock = threading.Lock()
        self._limit = limit

    def add(self, ticket: str, customer: str | None, result: dict) -> dict:
        with self._lock:
            tid = next(self._ids)
            record = {"id": tid, "customer": customer, "ticket": ticket,
                      "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
            self._items[tid] = record
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
            return record

    def get(self, tid: int) -> dict | None:
        return self._items.get(tid)

    def list(self, escalated: bool | None = None, team: str | None = None) -> list[dict]:
        items = reversed(list(self._items.values()))
        return [t for t in items
                if (escalated is None or t["routing"]["escalate"] == escalated)
                and (team is None or t["routing"]["team"] == team)]

    def stats(self) -> dict:
        items = list(self._items.values())
        by = lambda key: {k: sum(1 for t in items if key(t) == k) for k in sorted({key(t) for t in items})}  # noqa: E731
        return {
            "total": len(items),
            "escalated": sum(t["routing"]["escalate"] for t in items),
            "auto_resolved": sum(not t["routing"]["escalate"] for t in items),
            "by_priority": by(lambda t: t["priority"]["label"]),
            "by_team": by(lambda t: t["routing"]["team"]),
            "by_sentiment": by(lambda t: t["sentiment"]["label"]),
        }
