"""Output guardrails for LLM-generated text: knowledge answers and router clarifications.

Blocking checks come first (citation, size, secrets, prompt leak, language). Then redaction:
a sentence that contains a link, e-mail address, phone number or code which neither the trusted
evidence nor the customer's own messages contain is removed — the typical payload of an
indirect injection ("Gutscheincode GRATIS100") or of a hallucination. Payment data is masked.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

from customer_agent.application.guardrails.language import looks_german
from customer_agent.application.guardrails.patterns import CODE, URL
from customer_agent.domain.guardrails import GuardrailAction, GuardrailDecision
from customer_agent.domain.pii import EMAIL, PHONE, SECRET, mask_personal_data, valid_phone

MAX_ANSWER_CHARS = 2000
MAX_CLARIFICATION_CHARS = 300
SHOP_DOMAIN = "nordelektro.example"
# Eight consecutive words shared with a system prompt: German answers never share that many
# words with the English instructions by accident.
LEAK_NGRAM = 8
_UNIT = re.compile(r"(?<=[.!?])\s+|\n+")  # sentences and lines (lists have no full stops)


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-zäöüß]", "", text.casefold())


@dataclass(frozen=True)
class Evidence:
    """What an answer may repeat: trusted documents and the customer's own words. Tokens that
    occur in untrusted documents count as unverified even if the customer repeats them."""

    trusted: str = ""
    untrusted: str = ""
    customer: str = ""
    _folded: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("trusted", "untrusted", "customer"):
            self._folded[name] = _compact(getattr(self, name))

    def verifies(self, token: str) -> bool:
        """Compared without spaces and dashes: "ANC-700" is verified by "ANC 700"."""
        compact = _compact(token)
        if compact in self._folded["trusted"]:
            return True
        return compact in self._folded["customer"] and compact not in self._folded["untrusted"]


@lru_cache(maxsize=32)
def _ngrams(text: str) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return frozenset(tuple(words[i : i + LEAK_NGRAM]) for i in range(len(words) - LEAK_NGRAM + 1))


def leaks_prompt(text: str, prompt: str) -> bool:
    return bool(prompt) and not _ngrams(text).isdisjoint(_ngrams(prompt))


def _own_domain(host: str) -> bool:
    return host == SHOP_DOMAIN or host.endswith(f".{SHOP_DOMAIN}")


def unverified_tokens(text: str, evidence: Evidence) -> list[str]:
    tokens: list[str] = []
    for raw in URL.findall(text):
        url = raw.rstrip(".,;:!?")
        host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
        if not _own_domain(host) and url.casefold() not in evidence.trusted.casefold():
            tokens.append(url)
    for email in EMAIL.findall(text):
        if not _own_domain(email.rsplit("@", 1)[1].casefold()) and not evidence.verifies(email):
            tokens.append(email)
    phones = [p for p in PHONE.findall(text) if valid_phone(p)]
    tokens += [t for t in [*CODE.findall(text), *phones] if not evidence.verifies(t)]
    return tokens


def _blocking_reasons(text: str, prompt: str, max_chars: int) -> list[str]:
    reasons = []
    if len(text) > max_chars:
        reasons.append("answer_too_long")
    if SECRET.search(text):
        reasons.append("secret_in_output")
    if leaks_prompt(text, prompt):
        reasons.append("prompt_leak")
    if looks_german(text) is False:
        reasons.append("not_german")
    return reasons


def _redact(answer: str, evidence: Evidence) -> tuple[str, list[str]]:
    reasons: list[str] = []
    text = answer
    offending = [unit for unit in _UNIT.split(answer) if unverified_tokens(unit, evidence)]
    if offending:
        reasons.append("unverified_content")
        for unit in offending:
            text = text.replace(unit, "")
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()
    text, pii = mask_personal_data(text, kinds=("pii_iban", "pii_payment_card"))
    if pii:
        reasons.append("pii_in_output")
    return text, reasons


def check_output(
    answer: str,
    cited: Sequence[str],
    offered: Collection[str],
    *,
    prompt: str = "",
    evidence: Evidence | None = None,
) -> GuardrailDecision:
    """Every knowledge answer cites a document it was given (FR-11), stays German (LR-03),
    leaks nothing, and repeats links, codes and numbers only from trusted evidence."""
    reasons = [] if set(cited) & set(offered) else ["missing_citation"]
    reasons += _blocking_reasons(answer, prompt, MAX_ANSWER_CHARS)
    if reasons:
        return GuardrailDecision(stage="output", action=GuardrailAction.BLOCK, reasons=reasons)
    if evidence is None:
        return GuardrailDecision(stage="output", action=GuardrailAction.ALLOW)
    text, redactions = _redact(answer, evidence)
    if not text:
        return GuardrailDecision(stage="output", action=GuardrailAction.BLOCK, reasons=redactions)
    if redactions:
        return GuardrailDecision(
            stage="output", action=GuardrailAction.REDACT, reasons=redactions, content=text
        )
    return GuardrailDecision(stage="output", action=GuardrailAction.ALLOW)


def check_generated_text(text: str, *, prompt: str, customer: str) -> GuardrailDecision:
    """For short LLM texts shown verbatim, e.g. the router's clarification question. There is
    no evidence, so any link, code or number must come from the customer's own words."""
    reasons = _blocking_reasons(text, prompt, MAX_CLARIFICATION_CHARS)
    if unverified_tokens(text, Evidence(customer=customer)):
        reasons.append("unverified_content")
    action = GuardrailAction.BLOCK if reasons else GuardrailAction.ALLOW
    return GuardrailDecision(stage="output", action=action, reasons=reasons)
