"""Intent routing result and deterministic entity extraction."""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from customer_agent.domain.intents import Intent

# Order numbers are always extracted by code, never taken from the LLM (it may invent them).
_ORDER_ID = re.compile(r"\bNE[\s-]?(\d{6})\b", re.IGNORECASE)


def extract_order_id(text: str) -> str | None:
    """``"ne 100245"`` / ``"NE-100245"`` → ``"NE-100245"``; ``None`` if absent."""
    match = _ORDER_ID.search(text)
    return f"NE-{match.group(1)}" if match else None


class RoutingSource(StrEnum):
    RULE = "rule"  # decided by deterministic rules, no LLM call
    LLM = "llm"
    FALLBACK = "fallback"  # LLM output unusable; continue as a knowledge question


class RoutingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: Intent | None
    confidence: float = Field(ge=0.0, le=1.0)
    order_id: str | None = None
    # The customer refers to one concrete order of theirs (vs. a general "how does it work?").
    about_specific_order: bool = False
    # The customer asks about another person's orders or data, or to act for them.
    third_party: bool = False
    # The customer demands money, a refund, a voucher or a discount from the assistant (Day 6).
    money_demand: bool = False
    # Self-contained German question for retrieval ("Und in die Schweiz?" → full question).
    standalone_query: str
    clarification_question: str | None = None
    source: RoutingSource
