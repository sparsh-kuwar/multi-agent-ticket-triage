"""The four triage agents. Each one has a single job and returns a label, a confidence and its source.

Classification agents (intent, priority, sentiment) try the LLM first and fall back to rules.
The routing agent combines their outputs into an owning team, an escalation decision, a next
action and a first response.
"""

from . import llm, rules

TEAMS = {
    "Billing": "Payments Support",
    "Refund": "Payments Support",
    "Technical Issue": "Technical Support",
    "Account": "Account Support",
    "Shipping": "Shipping Support",
    "General": "General Support",
}

CONFIDENCE_FLOOR = 0.6  # below this, a human checks the classification

CLASSIFY_PROMPT = """You are the {role} agent in a customer-support triage system.
Classify the ticket into exactly one of: {labels}.
{guidance}
Ticket:
\"\"\"{ticket}\"\"\"

Reply with JSON only: {{"label": "<one of the labels>", "confidence": <0.0-1.0>, "evidence": ["<short quote>", ...]}}"""


def _classify(role: str, labels: list[str], guidance: str, ticket: str, fallback) -> dict:
    data = llm.ask_json(CLASSIFY_PROMPT.format(role=role, labels=", ".join(labels),
                                               guidance=guidance, ticket=ticket))
    if data:
        label = next((lbl for lbl in labels if lbl.lower() == str(data.get("label", "")).strip().lower()), None)
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = None
        if label and confidence is not None:
            evidence = [str(e) for e in data.get("evidence", [])][:5]
            return {"label": label, "confidence": round(confidence, 2), "evidence": evidence, "source": "llm"}
    return {**fallback(ticket), "source": "rules"}


def intent_agent(ticket: str) -> dict:
    return _classify("intent", rules.INTENTS, "Pick what the customer needs help with.",
                     ticket, rules.classify_intent)


def priority_agent(ticket: str) -> dict:
    return _classify("priority", rules.PRIORITIES,
                     "Consider security risk, financial impact, service disruption, number of users "
                     "affected and time sensitivity. URGENT means act within the hour.",
                     ticket, rules.classify_priority)


def sentiment_agent(ticket: str) -> dict:
    return _classify("sentiment", rules.SENTIMENTS, "Judge the customer's emotional tone.",
                     ticket, rules.classify_sentiment)


RESPONSE_TEMPLATES = {
    "Billing": "We're reviewing the charges on your account and will confirm the correct amount. "
               "If you were charged in error, we'll reverse it.",
    "Refund": "We've received your refund request and are checking it against the order. "
              "We'll confirm the outcome and the expected timeline.",
    "Technical Issue": "Our technical team is looking into the problem. If you can, reply with any error "
                       "message, the device or browser you use, and when it started.",
    "Account": "We're looking into your account. For your security we'll never ask for your password. "
               "Please use the official reset link if you need to change it.",
    "Shipping": "We're checking the status of your shipment with the courier and will update you "
                "with the latest tracking information.",
    "General": "Thanks for getting in touch. A member of our team will look at your message and reply.",
}

NEXT_ACTIONS = {
    "Billing": "Verify recent charges against the customer's plan; reverse duplicates.",
    "Refund": "Check order and refund eligibility; process or explain the decision.",
    "Technical Issue": "Reproduce the issue, check status page and logs; open an engineering ticket if needed.",
    "Account": "Verify identity, then restore access or update account details.",
    "Shipping": "Look up tracking with the courier; reship or refund if lost.",
    "General": "Answer the question or forward it to the right team.",
}

SLA_HOURS = {"URGENT": 1, "HIGH": 4, "MEDIUM": 24, "LOW": 72}


def escalation_reasons(intent: dict, priority: dict, sentiment: dict) -> list[str]:
    reasons = []
    if priority["label"] in ("URGENT", "HIGH"):
        reasons.append(f"{priority['label']} priority")
    if sentiment["label"] == "Angry":
        reasons.append("angry customer")
    for name, result in (("intent", intent), ("priority", priority), ("sentiment", sentiment)):
        if result["confidence"] < CONFIDENCE_FLOOR:
            reasons.append(f"low {name} confidence ({result['confidence']:.0%})")
    return reasons


def routing_agent(ticket: str, intent: dict, priority: dict, sentiment: dict) -> dict:
    team = TEAMS[intent["label"]]
    reasons = escalation_reasons(intent, priority, sentiment)
    escalate = bool(reasons)
    sla = SLA_HOURS[priority["label"]]

    if escalate:
        response = ("Thanks for contacting us. Your ticket has been passed to a senior member of our "
                    f"{team} team, who will reply within {sla} hour{'s' if sla != 1 else ''}.")
        action = f"Human review: {', '.join(reasons)}. Then: {NEXT_ACTIONS[intent['label']]}"
    else:
        response = RESPONSE_TEMPLATES[intent["label"]]
        action = NEXT_ACTIONS[intent["label"]]
    source = "rules"

    if not escalate:
        # The LLM may write a more specific first response; escalated tickets keep the fixed wording.
        data = llm.ask_json(
            "Write a short, polite first response (2-3 sentences) to this customer-support ticket. "
            f"Team: {team}. Intent: {intent['label']}. Do not promise refunds or timelines.\n"
            f'Ticket:\n"""{ticket}"""\n\nReply with JSON only: {{"response": "<text>"}}')
        if data and isinstance(data.get("response"), str) and data["response"].strip():
            response, source = data["response"].strip(), "llm"

    if sentiment["label"] in ("Frustrated", "Angry") and not response.startswith("Sorry"):
        response = "Sorry for the trouble. " + response

    return {
        "team": team,
        "escalate": escalate,
        "escalation_reasons": reasons,
        "sla_hours": sla,
        "next_action": action,
        "first_response": response,
        "auto_response_sent": not escalate,
        "source": source,
    }
