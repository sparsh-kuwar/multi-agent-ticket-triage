# Multi-Agent Customer Support Ticket Triage

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sparsh-kuwar/multi-agent-ticket-triage)
Live: https://multi-agent-ticket-triage.onrender.com
A Flask service that triages incoming support tickets. Four agents work in sequence:

1. **Intent agent** decides what the customer needs: Billing, Refund, Technical Issue, Account, Shipping or General.
2. **Priority agent** decides how fast the ticket needs attention: LOW, MEDIUM, HIGH or URGENT. It looks at security risk, money, service disruption and time pressure.
3. **Sentiment agent** judges the customer's tone: Positive, Neutral, Frustrated or Angry.
4. **Routing agent** combines the three results. It picks the owning team, sets a reply deadline, writes the next action and drafts a first response. It also decides whether a human must take over.

Each classification comes with a **confidence** and its **evidence** (the words that drove the decision).

## How escalation works

The routing agent sends a ticket to a human when any of these is true:

| Trigger | Example |
|---|---|
| Priority is HIGH or URGENT | "My account was hacked and there is an unauthorized payment" |
| Sentiment is Angry | "This is ridiculous, worst service ever!!!" |
| Any agent's confidence is below 60% | "hello" (no clear intent) |

- **Escalated tickets:** the customer gets a holding reply that names the team and the reply deadline. The next action starts with the reasons for escalation.
- **Other tickets:** the customer gets an automatic first response written for their intent.
- **Frustrated or angry customers:** the response starts with an apology.

| Priority | Reply within |
|---|---|
| URGENT | 1 hour |
| HIGH | 4 hours |
| MEDIUM | 24 hours |
| LOW | 72 hours |

## LLM mode and rule-based mode

- **LLM mode:** used when `OPENAI_API_KEY` is set. Each agent asks the model (through LangChain) for JSON with a label, a confidence and evidence. The routing agent also asks the model to write the first response for routine tickets.
- **Rule-based mode:** used when no key is set. It is also the **automatic fallback** whenever an LLM call fails, times out or returns something invalid. The rule-based agents score weighted keywords, handle simple negation ("not great"), and treat ALL-CAPS text as a sign of anger.

The page header and `GET /api/health` both show which mode is active.

## REST API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/tickets` | Triage a ticket and store it. Body: `{"ticket": "...", "customer": "optional"}`. Returns `201` |
| `POST` | `/api/triage` | Triage a ticket **without** storing it, for systems that keep their own records |
| `GET` | `/api/tickets` | List tickets, newest first. Filters: `?escalated=true\|false`, `?team=Payments Support` |
| `GET` | `/api/tickets/<id>` | Get one ticket |
| `GET` | `/api/stats` | Counts by priority, team, sentiment and escalation |
| `GET` | `/api/health` | Status and active mode |
| `GET` | `/` | Web UI: submit tickets, see each agent's decision, browse the queue |

Invalid input returns JSON errors: `400` for a missing JSON body or bad filters, `422` for an empty or over-long ticket (more than 5,000 characters), and `404` for an unknown ticket.

Example:

```sh
curl -s -X POST localhost:5000/api/tickets -H "Content-Type: application/json" \
  -d '{"ticket": "I was charged twice for my subscription", "customer": "Priya"}'
```

```json
{
  "id": 1007, "customer": "Priya", "ticket": "I was charged twice for my subscription",
  "intent":    {"label": "Billing", "confidence": 1.0, "evidence": ["charged", "charge", "subscription"], "source": "rules"},
  "priority":  {"label": "HIGH", "confidence": 0.67, "evidence": ["charged twice"], "source": "rules"},
  "sentiment": {"label": "Neutral", "confidence": 0.7, "evidence": [], "source": "rules"},
  "routing": {
    "team": "Payments Support", "escalate": true, "escalation_reasons": ["HIGH priority"],
    "sla_hours": 4, "next_action": "Human review: HIGH priority. Then: Verify recent charges ...",
    "first_response": "Thanks for contacting us. Your ticket has been passed to ...", "auto_response_sent": false
  },
  "mode": "rules"
}
```

## Run locally

Requires Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add OPENAI_API_KEY for LLM mode
flask --app app run --debug
```

Open http://127.0.0.1:5000.

## Tests

```sh
python -m pytest -q
```

There are 41 tests. They cover each classifier, the escalation rules, the LLM fallback (with a mocked client), input validation and every API endpoint. The tests never call a real LLM, even if `.env` contains a key.

## Deploy to Render

`render.yaml` is a Render Blueprint for one free web service.

1. In Render, choose **New → Blueprint** and select this repository.
2. Optional: when Render asks for `OPENAI_API_KEY`, enter a key to turn on LLM mode, or leave it empty to use rule-based mode.

Things to know:

- Tickets are stored **in memory**. They reset when the service restarts, and free services restart after about 15 minutes idle. Six sample tickets are reloaded on each start (`SEED_SAMPLE_TICKETS`).
- The service runs a single Gunicorn worker, so every request sees the same ticket store.

## Project layout

```
app.py               Flask app: REST API and web UI
triage/agents.py     Intent, priority, sentiment and routing agents; escalation rules
triage/rules.py      Rule-based classifiers (default and fallback)
triage/llm.py        Optional OpenAI client via LangChain
triage/pipeline.py   Runs the agents in order; in-memory ticket store
templates/, static/  Web UI
tests/               pytest suite
render.yaml          Render Blueprint
```
