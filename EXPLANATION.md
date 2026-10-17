# Multi-Agent Ticket Triage: interview guide

This guide is for explaining and defending the project in an interview. Each section names the file and function it describes.

Live demo: https://multi-agent-ticket-triage.onrender.com

## 1. The 30-second pitch

> A Flask service that triages support tickets with four specialised agents. Intent, priority and sentiment agents each classify the ticket and return a label, a confidence and the evidence words. A routing agent combines them to pick the team, set the reply deadline and either auto-reply or hand the ticket to a human. Each classifier tries an LLM first and falls back to deterministic rules on any failure, so the service keeps working without an API key. Escalation is decided by explicit rules, not by the LLM.

## 2. The 2-minute walkthrough

Code: `triage/pipeline.py`, function `triage`.

1. **Intent agent.** Billing / Refund / Technical Issue / Account / Shipping / General.
2. **Priority agent.** LOW / MEDIUM / HIGH / URGENT, judged on security risk, money, disruption, time pressure and legal or churn threats.
3. **Sentiment agent.** Positive / Neutral / Frustrated / Angry.
4. **Routing agent** (`agents.routing_agent`):
   - team from intent (`TEAMS`, e.g. Billing and Refund → Payments Support),
   - deadline from priority (`SLA_HOURS`: URGENT 1h, HIGH 4h, MEDIUM 24h, LOW 72h),
   - **escalate if** priority is HIGH/URGENT, **or** sentiment is Angry, **or** any agent's confidence < 0.60,
   - escalated → fixed holding reply naming the team and deadline, and a next action starting with the escalation reasons,
   - not escalated → auto first response (an LLM-written reply if available, otherwise a template),
   - Frustrated/Angry → the reply starts with "Sorry for the trouble."
5. **Storage** (`TicketStore`). Thread-safe, bounded (500), in-memory.

## 3. How each classifier works

### LLM path (`agents._classify`, `llm.ask_json`)
One shared prompt template per role: the allowed labels, guidance, and "reply with JSON only: label, confidence, evidence". The output is **validated**: the label must match an allowed label (case-insensitive), and the confidence must parse as a float and is clamped to 0–1. Anything else → rule fallback. `ask_json` returns `None` on *any* exception (network, auth, rate limit, bad JSON), never an error.

### Rule path (`triage/rules.py`)
- **Weighted keywords** per label (e.g. "billing" 3, "card" 1). The highest-scoring label wins.
- **Word boundaries** for short words (`_matches`), so "app" doesn't match "happy" and "500" doesn't match "1500".
- **Heuristic confidence** (`_confidence`): `0.5 + 0.3·strength + 0.2·margin`, where strength = score/6 (capped at 1) and margin = (best − runner-up)/best. A strong, clear winner → about 1.0. No signal → 0.4, which is below the 0.6 floor, so a human checks it.
- **Priority** is a sum of urgency signals minus "no rush" signals: ≥6 URGENT, ≥3 HIGH, ≥1 MEDIUM. No signals → MEDIUM at 0.65 confidence ("a normal ticket" is itself weak evidence).
- **Sentiment** extras: negated positives ("not great") don't count as positive, and **ALL CAPS** (>60% uppercase, >20 letters) adds +3 to Angry.

The module docstring says it directly: confidence here *"is not a calibrated probability"*. Say that before you're asked.

## 4. Key design decisions and why

| Decision | Why |
|---|---|
| Separate agents per dimension | Each has one job, its own prompt and its own fallback. You can improve or replace one without touching the others, and each decision is explainable on its own. |
| LLM-first, rules fallback | LLMs handle phrasing that rules miss. Rules guarantee the service never goes down because of the LLM, and make it free to demo. |
| **Escalation is rule-based, not LLM-based** | Who reaches a human is a business policy. It must be predictable, auditable and testable. |
| Low confidence → human | The system knows when it doesn't know. "hello" (no clear intent) goes to a person, not an auto-reply. |
| Escalated tickets get a fixed reply | No LLM-written promises to an angry or high-risk customer. The LLM is told "do not promise refunds or timelines" even for auto replies. |
| Evidence on every decision | An agent can see *why*: "HIGH because 'charged twice'". This builds trust and helps debugging. |
| `temperature=0`, timeout 20s, 1 retry | Repeatable labels, bounded latency |
| `POST /api/triage` (no storage) and `POST /api/tickets` (stores) | Lets another helpdesk system call it as a pure classifier |
| One Gunicorn worker | The store is in-memory. Several workers would each have a different store. |
| Lazy LangChain import | Rule-based mode doesn't need the dependency loaded |

## 5. Code map

| File | What it does |
|---|---|
| `app.py` | Flask app factory, REST endpoints, input validation (400/422/404), sample tickets |
| `triage/agents.py` | The four agents, team and deadline tables, escalation rules, reply templates |
| `triage/rules.py` | Keyword tables and rule classifiers |
| `triage/llm.py` | Optional OpenAI client through LangChain, JSON extraction, failure → None |
| `triage/pipeline.py` | Runs the agents in order, times them, `TicketStore` |
| `tests/test_triage.py` | 41 tests (README): classifiers, escalation, LLM fallback with a mocked client, validation, every endpoint. Never calls a real LLM. |

## 6. Weaknesses to state openly

1. **"Multi-agent" means a fixed pipeline of specialised classifiers**, not autonomous agents that plan or talk to each other. Say this yourself. It is a deliberate choice for predictability.
2. **Agents run sequentially.** Intent, priority and sentiment are independent, so in LLM mode they could run in parallel (thread pool or async) and cut latency to about a third. Or one LLM call could return all three.
3. **Rule confidence isn't calibrated**, and LLM confidence is self-reported.
4. **Keyword rules are brittle**: sarcasm ("great, broken again"), typos, and languages other than English. Negation handling only covers positive words.
5. **No evaluation set.** I can't quote accuracy. A labelled set of a few hundred tickets with per-label precision/recall would be the first addition.
6. **In-memory storage** is lost on restart, with one worker only. Production needs Postgres or Redis.
7. **"evidence" from the LLM is not checked** against the ticket text. It could quote words that aren't there. A substring check would catch that.
8. **The LLM sees raw ticket text**, so prompt injection could steer a label. The impact is limited because escalation rules, the label whitelist and fixed escalation replies all sit outside the LLM.
9. **No auth or rate limiting** on the public API.

## 7. Likely interview questions

**Q: Why four agents instead of one LLM call?**
Separation of concerns: each agent has a focused prompt, a validated output and an independent rule fallback, and each can be tested alone. The trade-off is three LLM calls instead of one. In production I would consider one call returning all three, while keeping the per-field validation and fallback.

**Q: Why is escalation not decided by the LLM?**
It is a policy decision with customer and business risk. Rules ("HIGH priority or angry or low confidence → human") are transparent, consistent and covered by tests. The LLM informs the inputs, the rules make the call.

**Q: How does the fallback work?**
`ask_json` catches every exception and returns `None`. `_classify` also rejects labels outside the allowed set and confidences that don't parse. In every one of those cases the rule classifier answers and the result is tagged `source: "rules"`, so you can see which path was used.

**Q: Why a confidence floor of 0.6?**
It's a trade-off between automation rate and risk. Higher → more tickets to humans, fewer wrong auto-replies. I would tune it on labelled data by plotting the auto-resolution rate against the error rate.

**Q: "My account was hacked and there's an unauthorized payment". Walk me through it.**
Intent: Account ("hacked" 3, "account" 2). Priority: "hacked" 4 + "unauthorized" 4 = 8 ≥ 6 → URGENT. Escalate (URGENT priority), Account Support, reply within 1 hour, fixed holding reply.

**Q: How would you scale it?**
A persistent DB, several workers or containers behind a load balancer, a queue for spikes, parallel or merged LLM calls, caching of near-duplicate tickets, and per-client API keys with rate limits.

**Q: How would you know it's working in production?**
Track the human override rate (how often agents change the team or priority), the auto-reply reopen rate, the SLA hit rate and the escalation rate. Overrides become labelled data for improving rules and prompts.

**Q: How is this relevant to Bruviti?**
Bruviti's contact-center and service products face the same problem: classify inbound requests, route them, and decide automate vs. human. The pattern here, specialised classifiers with evidence, a safe fallback and rule-based escalation, transfers directly to equipment service requests.

## 8. What I would build next

1. A labelled evaluation set and a per-label precision/recall report.
2. Parallel classifier calls, or a single structured LLM call.
3. A check that LLM evidence quotes appear in the ticket.
4. Persistent storage, auth and rate limiting.
5. Agent override feedback stored and used to retune keyword weights and prompts.
6. Multilingual support (Hindi and other regional languages for Indian customers).
