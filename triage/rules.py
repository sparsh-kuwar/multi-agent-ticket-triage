"""Rule-based classifiers: the default when no LLM is configured, and the fallback when an LLM call fails.

Each classifier scores every label by weighted keyword matches and returns the best label with a
confidence between 0 and 1. Confidence is a heuristic: it grows with the winning score and with the
margin over the runner-up. It is not a calibrated probability.
"""

import re

INTENTS = ["Billing", "Refund", "Technical Issue", "Account", "Shipping", "General"]
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "URGENT"]
SENTIMENTS = ["Positive", "Neutral", "Frustrated", "Angry"]

INTENT_KEYWORDS = {
    "Billing": {"invoice": 2, "charged": 2, "charge": 2, "billing": 3, "bill": 2, "payment": 2,
                "subscription": 2, "card": 1, "overcharged": 3, "double charged": 3, "pricing": 1, "plan": 1},
    "Refund": {"refund": 3, "money back": 3, "reimburse": 3, "return": 1, "cancel order": 2, "chargeback": 3},
    "Technical Issue": {"error": 2, "bug": 2, "crash": 3, "crashes": 3, "not working": 2, "broken": 2,
                        "down": 2, "outage": 3, "slow": 1, "fails": 2, "failed": 2, "500": 2, "timeout": 2,
                        "app": 1, "website": 1, "login page": 1, "api": 1, "sync": 1},
    "Account": {"password": 3, "login": 2, "log in": 2, "sign in": 2, "account": 2, "locked": 3,
                "2fa": 3, "two-factor": 3, "email change": 2, "username": 2, "delete my account": 3, "hacked": 3},
    "Shipping": {"delivery": 3, "delivered": 2, "shipping": 3, "shipment": 3, "package": 2, "parcel": 2,
                 "tracking": 3, "courier": 3, "arrived": 2, "late": 1, "lost": 1, "order": 1},
    "General": {"question": 1, "how do i": 1, "information": 1, "feedback": 1, "suggestion": 1},
}

# Signals that raise priority. Each match adds its weight; the total maps to a level.
PRIORITY_SIGNALS = {
    # security and safety
    "hacked": 4, "unauthorized": 4, "fraud": 4, "data breach": 4, "security": 3, "stolen": 3,
    # service disruption
    "outage": 3, "down": 2, "cannot access": 2, "can't access": 2, "not working": 1, "production": 2,
    "all users": 3, "entire team": 2, "blocked": 2, "crash": 1,
    # money
    "double charged": 3, "charged twice": 3, "overcharged": 2, "chargeback": 3, "refund": 1,
    # time pressure
    "urgent": 3, "asap": 2, "immediately": 2, "right now": 2, "today": 1, "deadline": 2, "emergency": 3,
    # legal or churn threats
    "lawyer": 3, "legal": 3, "cancel my subscription": 2, "switching to": 2,
}
LOW_SIGNALS = {"question": 1, "just wondering": 2, "feedback": 2, "suggestion": 2, "no rush": 3,
               "when you get a chance": 3, "curious": 2}

SENTIMENT_KEYWORDS = {
    "Positive": {"thank": 2, "thanks": 2, "great": 2, "love": 2, "appreciate": 2, "awesome": 2, "happy": 2},
    "Frustrated": {"still": 1, "again": 1, "frustrat": 3, "annoying": 2, "disappointed": 2, "waiting": 1,
                   "multiple times": 2, "third time": 2, "no response": 2, "unacceptable": 2, "confusing": 1},
    "Angry": {"ridiculous": 3, "terrible": 3, "worst": 3, "scam": 3, "furious": 3, "angry": 3, "lawyer": 2,
              "useless": 2, "pathetic": 3, "never again": 3, "!!!": 2, "fed up": 3},
}

NEGATIONS = ("not ", "no ", "never ", "n't ")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _matches(text: str, phrase: str) -> bool:
    if phrase.isalnum() and len(phrase) <= 5:
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
    return phrase in text


def _score(text: str, keywords: dict[str, int]) -> tuple[int, list[str]]:
    hits = [k for k in keywords if _matches(text, k)]
    return sum(keywords[k] for k in hits), hits


def _confidence(best: int, runner_up: int) -> float:
    if best == 0:
        return 0.4
    strength = min(best / 6, 1.0)
    margin = (best - runner_up) / best
    return round(0.5 + 0.3 * strength + 0.2 * margin, 2)


def classify_intent(ticket: str) -> dict:
    text = _normalise(ticket)
    scored = {label: _score(text, kw) for label, kw in INTENT_KEYWORDS.items()}
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    (label, (best, hits)), (_, (second, _)) = ranked[0], ranked[1]
    if best == 0:
        label = "General"
    return {"label": label, "confidence": _confidence(best, second), "evidence": hits}


def classify_priority(ticket: str) -> dict:
    text = _normalise(ticket)
    up, up_hits = _score(text, PRIORITY_SIGNALS)
    down, down_hits = _score(text, LOW_SIGNALS)
    score = up - down
    if score >= 6:
        label = "URGENT"
    elif score >= 3:
        label = "HIGH"
    elif score >= 1:
        label = "MEDIUM"
    elif down > 0:
        label = "LOW"
    else:
        label = "MEDIUM"  # no urgency signal: a normal ticket
    evidence = up_hits + down_hits
    # No urgency words is itself weak evidence of a normal ticket, so confidence stays above the review floor.
    confidence = 0.65 if not evidence else round(min(0.6 + 0.07 * len(evidence), 0.95), 2)
    return {"label": label, "confidence": confidence, "evidence": evidence}


def classify_sentiment(ticket: str) -> dict:
    text = _normalise(ticket)
    scored = {label: _score(text, kw) for label, kw in SENTIMENT_KEYWORDS.items()}
    # "not great", "never happy": a negated positive word does not count as positive.
    if any(neg + word in text for neg in NEGATIONS for word in SENTIMENT_KEYWORDS["Positive"]):
        scored["Positive"] = (0, [])
    # Shouting adds to anger.
    letters = [c for c in ticket if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        s, h = scored["Angry"]
        scored["Angry"] = (s + 3, h + ["ALL CAPS"])
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    (label, (best, hits)), (_, (second, _)) = ranked[0], ranked[1]
    if best == 0:
        return {"label": "Neutral", "confidence": 0.7, "evidence": []}
    return {"label": label, "confidence": _confidence(best, second), "evidence": hits}
