"""Output guardrails for LLM-generated text: knowledge answers and router clarifications.

Blocking checks come first (citation, size, secrets, prompt leak, language). Then redaction:
a sentence that contains a link, e-mail address, phone number or code which neither the trusted
evidence nor the customer's own messages contain is removed — the typical payload of an
indirect injection ("Gutscheincode GRATIS100") or of a hallucination. So is a sentence that
offers to forward the request or connect to a person, or claims that it was forwarded: only the
state machine asks that, and only a ticket that exists may be announced (ADR-0014). Payment data
is masked.

``check_output`` runs the steps at once; the answer agent's middleware runs them one hook each
(rebuild phase 2), so both take the same path.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
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
# The model copies the assistant's earlier ticket question from the history, offers a handoff
# of its own accord ("Gern verbinde ich Sie mit einem Mitarbeiter"), or announces one that never
# happened ("Ich gebe Ihre Defektmeldung weiter", "Sie erhalten eine Ticketnummer").
HANDOFF_OFFER = re.compile(
    r"\b(weitergeben|weiterleiten|verbinden)\?|verbinde ich sie"
    r"|mit einem (mitarbeiter|kollegen|menschen) (sprechen|verbinden)\?"
    r"|\b(ich|wir) (gebe|geben|leite|leiten)\b[^.!?\n]{0,80}\bweiter\b"
    r"|\b(weitergeleitet|weitergegeben)\b|\bticket"
    r"|\b(kann|könnte|würde) ich\b[^.!?\n]{0,80}\b(weiterleiten|weitergeben|verbinden|übergeben)\b",
    re.IGNORECASE,
)


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


def _remove_unverified(answer: str, evidence: Evidence) -> tuple[str, list[str]]:
    reasons: list[str] = []
    text = answer
    units = _UNIT.split(answer)
    offending = [unit for unit in units if unverified_tokens(unit, evidence)]
    if offending:
        reasons.append("unverified_content")
    offers = [unit for unit in units if HANDOFF_OFFER.search(unit)]
    if offers:
        reasons.append("handoff_offer")
    for unit in offending + offers:
        text = text.replace(unit, "")
    if offending or offers:
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()
    return text, reasons


@dataclass(frozen=True)
class OutputScreen:
    """The check so far: the answer without citation markers (redacted, then masked) and the
    verdict."""

    text: str
    reasons: tuple[str, ...] = ()
    blocked: bool = False


def screen_output(
    answer: str,
    cited: Sequence[str],
    offered: Collection[str],
    *,
    prompt: str = "",
    evidence: Evidence | None = None,
) -> OutputScreen:
    """Every knowledge answer cites a document it was given (FR-11), stays German (LR-03),
    leaks nothing, and repeats links, codes and numbers only from trusted evidence: sentences
    that do not are removed; nothing left blocks the answer."""
    reasons = [] if set(cited) & set(offered) else ["missing_citation"]
    reasons += _blocking_reasons(answer, prompt, MAX_ANSWER_CHARS)
    if reasons:
        return OutputScreen(text=answer, reasons=tuple(reasons), blocked=True)
    if evidence is None:
        return OutputScreen(text=answer)
    text, redactions = _remove_unverified(answer, evidence)
    return OutputScreen(text=text, reasons=tuple(redactions), blocked=not text)


def mask_output(screen: OutputScreen) -> OutputScreen:
    """IBAN and card numbers masked; a blocked answer is never shown, so it stays as it is."""
    if screen.blocked:
        return screen
    text, pii = mask_personal_data(screen.text, kinds=("pii_iban", "pii_payment_card"))
    reasons = (*screen.reasons, "pii_in_output") if pii else screen.reasons
    return replace(screen, text=text, reasons=reasons)


def output_decision(screen: OutputScreen) -> GuardrailDecision:
    """``content`` only for a redacted answer: a blocked one is replaced by a fallback."""
    reasons = list(screen.reasons)
    if screen.blocked:
        return GuardrailDecision(stage="output", action=GuardrailAction.BLOCK, reasons=reasons)
    if reasons:
        return GuardrailDecision(
            stage="output", action=GuardrailAction.REDACT, reasons=reasons, content=screen.text
        )
    return GuardrailDecision(stage="output", action=GuardrailAction.ALLOW)


def check_output(
    answer: str,
    cited: Sequence[str],
    offered: Collection[str],
    *,
    prompt: str = "",
    evidence: Evidence | None = None,
) -> GuardrailDecision:
    screen = screen_output(answer, cited, offered, prompt=prompt, evidence=evidence)
    return output_decision(mask_output(screen))


def check_generated_text(text: str, *, prompt: str, customer: str) -> GuardrailDecision:
    """For short LLM texts shown verbatim, e.g. the router's clarification question. There is
    no evidence, so any link, code or number must come from the customer's own words."""
    reasons = _blocking_reasons(text, prompt, MAX_CLARIFICATION_CHARS)
    if unverified_tokens(text, Evidence(customer=customer)):
        reasons.append("unverified_content")
    action = GuardrailAction.BLOCK if reasons else GuardrailAction.ALLOW
    return GuardrailDecision(stage="output", action=action, reasons=reasons)
