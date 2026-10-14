"""Guardrails (Day 3 baseline, Day 6 layers, ADR-0011). Pure checks return a
``GuardrailDecision``; the orchestrator decides what the customer sees.

- ``input``: normalisation, size, prompt-injection heuristics, personal data masking
- ``classifier``: LLM second opinion on prompt attacks (import it from its module)
- ``evidence``: untrusted documents quoted as data
- ``output``: citation, size, secrets, prompt leak, language, unverifiable tokens, payment data
"""

from customer_agent.application.guardrails.input import (
    MAX_INPUT_CHARS,
    PAYMENT_DATA_REASONS,
    check_input,
)
from customer_agent.application.guardrails.output import (
    MAX_ANSWER_CHARS,
    Evidence,
    check_generated_text,
    check_output,
)

__all__ = [
    "MAX_ANSWER_CHARS",
    "MAX_INPUT_CHARS",
    "PAYMENT_DATA_REASONS",
    "Evidence",
    "check_generated_text",
    "check_input",
    "check_output",
]
