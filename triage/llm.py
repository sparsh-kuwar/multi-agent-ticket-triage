"""Optional LLM backend (OpenAI through LangChain).

The LLM is used only when OPENAI_API_KEY is set and TRIAGE_USE_LLM is not "false".
Every call returns parsed JSON or None; callers fall back to the rule-based classifiers on None.
"""

import json
import logging
import os
import re

log = logging.getLogger(__name__)

_llm = None


def enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and os.environ.get("TRIAGE_USE_LLM", "true").lower() != "false"


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _client():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI  # imported lazily: not needed in rule-based mode

        _llm = ChatOpenAI(model=model_name(), temperature=0,
                          timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "20")), max_retries=1)
    return _llm


def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def ask_json(prompt: str) -> dict | None:
    """Send one prompt that asks for a JSON object. Returns the object, or None on any failure."""
    if not enabled():
        return None
    try:
        reply = _client().invoke(prompt)
        data = _extract_json(reply.content)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # network, auth, rate limit, bad JSON: all fall back to rules
        log.warning("LLM call failed, using rule-based fallback: %s", exc)
        return None
