"""Guardrail decision contract (docs/architecture.md §5.1). Guardrails judge; the orchestrator
decides what the customer sees."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GuardrailAction(StrEnum):
    ALLOW = "allow"  # pass content through unchanged
    REDACT = "redact"  # pass through the modified content in ``content``
    BLOCK = "block"  # discard content; the orchestrator answers with a fallback


class GuardrailDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["input", "output"]
    action: GuardrailAction
    reasons: list[str] = Field(default_factory=list)
    content: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action is GuardrailAction.BLOCK


# Reason codes of guardrail interventions, shared with the evaluation: a benign case whose
# reply was blocked with one of these is a guardrail false positive (NFR-08).
BLOCK_REASONS = frozenset(
    {
        "prompt_injection",
        "prompt_injection_llm",
        "input_too_long",
        "missing_citation",
        "answer_too_long",
        "secret_in_output",
        "prompt_leak",
        "not_german",
        "unverified_content",
    }
)
# The reply was delivered, but changed (sentences removed, values masked).
REDACTION_REASONS = frozenset({"unverified_content", "pii_in_output"})
