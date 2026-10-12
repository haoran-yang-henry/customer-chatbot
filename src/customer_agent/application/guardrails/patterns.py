"""Detection patterns shared by the input, evidence and output guardrails (ADR-0011).

Deterministic and explainable: every injection hit has a signal id that appears in logs and
evaluation reports. Patterns cover German and English, because attacks against a German
assistant are often written in English. They are a first line, not the only one: paraphrases
they miss are left to the LLM classifier and to the layers behind it.
"""

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Signal:
    id: str
    weight: float
    pattern: re.Pattern[str]


def _p(regex: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(regex, flags)


# A message is blocked at a total weight of 1.0. Weight 1.0 = unambiguous on its own; weaker
# signals (words with an ordinary meaning, such as "Administrator") must co-occur.
BLOCK_THRESHOLD = 1.0
# Untrusted documents are held to a stricter standard: losing a line of a forum post is cheap.
UNTRUSTED_LINE_THRESHOLD = 0.5

INJECTION_SIGNALS: tuple[Signal, ...] = (
    # Imperatives only: "Ich habe vergessen, welche Regeln gelten" is a normal question.
    Signal(
        "override",
        1.0,
        _p(
            r"\b(ignorier(e|en sie)?|vergiss|vergesst|vergessen sie|missachte|überschreibe|"
            r"umgehe|verwirf)\b.{0,40}?\b(anweisung|instruktion|regel|vorgabe|richtlinie|"
            r"prompt|befehl|einschränkung|beschränkung)"
        ),
    ),
    Signal(
        "override",
        1.0,
        _p(r"\b(ignoriere|vergiss)\s+(alles|das)\s+(bisher|vorher|oben|zuvor|davor)"),
    ),
    Signal(
        "override",
        1.0,
        _p(
            r"\b(ignore|disregard|forget|override|bypass)\b(\s+\S+){0,5}?\s+(instructions?|"
            r"rules?|guidelines?|prompts?|directives?|polic(y|ies)|guardrails?|restrictions?)\b"
        ),
    ),
    Signal(
        "prompt_extraction",
        1.0,
        _p(
            r"\b(zeig\w*|gib|nenn\w*|verrat\w*|wiederhole|print|show|reveal|repeat|output|"
            r"display|tell me)\b.{0,40}?\b(system\s?-?prompt|systemanweisung|systemnachricht|"
            r"deine?n? (anweisungen|instruktionen|regeln|vorgaben|prompt)|"
            r"your (instructions|rules|system prompt|prompt|guidelines)|initial prompt|"
            r"hidden instructions)"
        ),
    ),
    Signal(
        "prompt_extraction",
        1.0,
        _p(
            r"\b(system\s?-?prompt|systemanweisung|deine anweisungen|your instructions)\b.{0,30}?"
            r"\b(aus|ausgeben|zeigen|nennen|verraten|wiederholen|preisgeben|reveal|print|show)\b"
        ),
    ),
    Signal("prompt_mention", 0.5, _p(r"\b(system\s?-?prompt|systemanweisung|prompt injection)")),
    Signal(
        "jailbreak",
        1.0,
        _p(
            r"\b(do anything now|developer mode|dev mode|entwicklermodus|jailbreak\w*|"
            r"god ?mode|unrestricted mode|uneingeschränkte[nr]? modus)\b"
        ),
    ),
    Signal("jailbreak", 1.0, re.compile(r"\bDAN\b")),  # case-sensitive: "Dan" is a name
    Signal(
        "role_override",
        0.6,
        _p(
            r"\b(du bist (jetzt|ab sofort|nun|ab jetzt)|ab (jetzt|sofort) bist du|"
            r"you are now|from now on,? you|act as an?|pretend (to be|you are)|tu so,? als|"
            r"spiele? die rolle|stell dir vor,? du bist|imagine you are|roleplay|rollenspiel)\b"
        ),
    ),
    Signal(
        "no_rules",
        0.6,
        _p(
            r"\b(keine|ohne|no|without)\s+(regeln|einschränkungen|beschränkungen|filter|zensur|"
            r"rules|restrictions|limits|filters)\b"
        ),
    ),
    Signal(
        "fake_role",
        1.0,
        _p(
            r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|assistant\|>|\[/?INST\]|<<SYS>>|"
            r"</?system>|\bBEGIN SYSTEM\b|###\s*(system|instruction|anweisung))"
        ),
    ),
    Signal(
        "fake_role",
        0.6,
        _p(
            r"((^|\n)\s*(system|assistant|assistent|developer|admin|administrator|operator)\s*:"
            r"|\[(system|admin|developer|assistant|assistent)\])",
        ),
    ),
    Signal(
        "new_rules",
        0.6,
        _p(
            r"\b(neue|new|updated|geänderte)\s+(anweisung|instruktion|instruction|regel|rule|"
            r"richtlinie|policy|systemanweisung)"
        ),
    ),
    Signal(
        "privilege",
        0.6,
        _p(
            r"\b(ich bin|der nutzer ist|der benutzer ist|der kunde ist|user is|i am|i'm)\b"
            r".{0,20}?\b(admin\w*|entwickler\w*|developer|root|superuser|moderator|"
            r"ihr chef|dein chef|your (boss|developer|creator))\b"
        ),
    ),
    Signal(
        "privilege",
        0.6,
        _p(
            r"\b(admin|administrator|debug|wartungs|maintenance|god|sudo)[- ]?(modus|mode|"
            r"rechte|zugriff|access|rights)\b"
        ),
    ),
    # No word boundary before the noun: German compounds ("internen Kulanzregeln"). The
    # override signal keeps its boundary on purpose: "Ignorieren Sie die Lieferanweisung" is a
    # normal request about a delivery note.
    Signal(
        "internal_info",
        0.5,
        _p(
            r"\bintern\w*\b.{0,30}?(richtlinie|dokument|regel|anweisung|information|daten|"
            r"code|prozess|eskalation)"
        ),
    ),
    Signal(
        "exfiltration",
        0.6,
        _p(
            r"\b(alle|sämtliche|all|every|liste)\b.{0,30}?\b(kunden|customers?|nutzer|users?|"
            r"benutzer|käufer)\w*\b.{0,30}?\b(e-?mail|mail|adressen|addresses|daten|data|"
            r"bestellungen|orders|namen|names|telefon|phone|passw)"
        ),
    ),
    Signal(
        "sql",
        1.0,
        _p(
            r"(\b(select\s+\*?\s*from|drop\s+table|union\s+select|insert\s+into)\b|"
            r"['\"]\s*or\s*['\"]?\w+['\"]?\s*=\s*['\"]?\w+)"
        ),
    ),
    Signal(
        "encoding_hint",
        0.5,
        _p(
            r"\b(base\s?64|rot-?13|hexadezimal|dekodier\w*|decode|entschlüssel\w*|decrypt|"
            r"rückwärts gelesen|reversed text)\b"
        ),
    ),
    Signal(
        "translate_follow",
        1.0,
        _p(
            r"\b(übersetz\w*|translate)\b.{0,120}?\b(befolg\w*|führ\w* (sie|es|das|ihn) aus|"
            r"ausführ\w*|follow|execute|obey|do what it says)"
        ),
    ),
    Signal(
        "language_override",
        0.5,
        _p(
            r"\b(antworte|answer|respond|reply|schreib)\w*\b.{0,30}?\b(nur noch|only|"
            r"ausschließlich|ab jetzt|from now on)\b.{0,20}?\b(englisch|english|französisch|"
            r"french|spanisch|spanish|russisch|russian|chinesisch|chinese)\b"
        ),
    ),
    # Typical of indirect injection planted in documents: "Hinweis für KI-Assistenten: ..."
    Signal(
        "ai_addressed",
        1.0,
        _p(
            r"\b(hinweis|nachricht|anweisung|botschaft|note|message|instruction)\w*\s+(für|an|to|"
            r"for)\s+(die |den |das |the |alle |all )?(ki|ai|künstliche intelligenz|"
            r"assistent\w*|assistant|chat-?bot\w*|sprachmodell\w*|language models?|llms?)\b"
        ),
    ),
    Signal(
        "broadcast",
        0.6,
        _p(
            r"\b(teile|sage|sag|erzähle|informiere|tell|inform)\w*\s+(jedem|allen|jeden|every|"
            r"all)\s+(kunden|customers?|nutzern|users?|käufern)\b"
        ),
    ),
)

# --- normalisation ------------------------------------------------------------------------

_TAG_BLOCK = range(0xE0000, 0xE0080)  # Unicode tags: invisible copies of ASCII ("ASCII smuggling")
_LEET = str.maketrans("0134578@$", "oieastbas")
_SPACED_OUT = re.compile(r"\b(?:\w[ .\-_*]){4,}\w\b")
_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")


def _invisible(char: str) -> bool:
    category = unicodedata.category(char)
    return category == "Cf" or (category == "Cc" and char not in "\n\t")


class Normalized(NamedTuple):
    text: str  # NFKC, invisible characters removed: the only version that reaches the LLM
    hidden: str  # ASCII that was hidden in Unicode tag characters
    removed: int  # number of invisible characters dropped


def normalize(text: str) -> Normalized:
    hidden = "".join(chr(ord(c) - 0xE0000) for c in text if ord(c) in _TAG_BLOCK)
    nfkc = unicodedata.normalize("NFKC", text)
    clean = "".join(c for c in nfkc if not _invisible(c))
    return Normalized(clean.strip(), hidden, len(nfkc) - len(clean))


def _variants(text: str) -> Iterable[str]:
    """The text as written plus de-obfuscated readings (leetspeak, s p a c e d letters)."""
    yield text
    collapsed = _SPACED_OUT.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group(0)), text)
    if collapsed != text:
        yield collapsed
    leet = text.translate(_LEET)
    if leet != text:
        yield leet


def injection_signals(text: str) -> dict[str, float]:
    """Signal id → weight for every signal that fires (each id counts once)."""
    hits: dict[str, float] = {}
    for variant in _variants(text):
        for signal in INJECTION_SIGNALS:
            if signal.pattern.search(variant):
                hits[signal.id] = max(hits.get(signal.id, 0.0), signal.weight)
    for run in _BASE64_RUN.findall(text):
        if sum(c.isdigit() for c in run) >= 2:  # long German compounds are letters only
            hits["encoded_blob"] = 0.6
    return hits


def injection_score(hits: dict[str, float]) -> float:
    return round(sum(hits.values()), 2)


# --- personal data and secrets --------------------------------------------------------------

# Candidates may run into the following words or digits ("... 12/27"), so the masks look for the
# longest valid prefix instead of trusting the regex boundaries.
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}", re.IGNORECASE)
_CARD = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,22}")
_CARD_PREFIX = re.compile(r"^(4|5[1-5]|2[2-7]|3[47]|6011|65)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# German and international numbers: leading 0, +CC or 00CC; 8-15 digits (E.164). Parcel tracking
# numbers (12-20 digits without a leading 0) and order numbers do not match.
PHONE = re.compile(r"(?<![\w-])(?:(?:\+|00)[1-9]\d{0,2}|0)[\d ()/-]{6,}\d(?![\w-])")
SECRET = re.compile(
    r"(\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bAKIA[0-9A-Z]{16}\b|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


# IBAN length per country (ISO 13616) for the countries customers pay from. The exact length
# matters: every prefix of a candidate is tried, and mod 97 alone passes one string in 97.
_IBAN_LENGTHS = {
    "DE": 22, "AT": 20, "CH": 21, "LI": 21, "LU": 20, "NL": 18, "BE": 16, "FR": 27, "IT": 27,
    "ES": 24, "PL": 28, "CZ": 24, "DK": 18, "SE": 24, "FI": 18, "IE": 22, "GB": 22, "PT": 25,
}  # fmt: skip


def valid_iban(candidate: str) -> bool:
    """Known country, exact length and ISO 13616 check digits (mod 97)."""
    compact = candidate.replace(" ", "").upper()
    if _IBAN_LENGTHS.get(compact[:2]) != len(compact):
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def valid_card(candidate: str) -> bool:
    """Luhn checksum plus a common issuer prefix: Luhn alone accepts one in ten numbers, which
    would mask parcel tracking numbers."""
    digits = re.sub(r"\D", "", candidate)
    if not 13 <= len(digits) <= 19 or not _CARD_PREFIX.match(digits):
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def valid_phone(candidate: str) -> bool:
    return 8 <= sum(c.isdigit() for c in candidate) <= 15


def _longest_valid_prefix(candidate: str, valid: Callable[[str], bool]) -> int:
    """Length of the longest prefix ending in a letter or digit that ``valid`` accepts."""
    for end in range(len(candidate), 0, -1):
        if candidate[end - 1].isalnum() and valid(candidate[:end]):
            return end
    return 0


@dataclass(frozen=True)
class _Validator:
    check: Callable[[str], bool]
    # IBANs and card numbers are often followed by more digits ("... 12/27"), so their longest
    # valid prefix counts. A phone number must be valid as a whole: a prefix of a 20-digit
    # parcel number is not a phone number.
    prefixes: bool


def _mask(text: str, pattern: re.Pattern[str], valid: _Validator | None, mask: str) -> str:
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        candidate = match.group(0)
        if valid is None:
            length = len(candidate)
        elif valid.prefixes:
            length = _longest_valid_prefix(candidate, valid.check)
        else:
            length = len(candidate) if valid.check(candidate) else 0
        if length:
            parts += [text[last : match.start()], mask]
            last = match.start() + length
    return "".join([*parts, text[last:]])


# Placeholders are German: they become part of the text the LLM and the transcript see.
_MASKS: tuple[tuple[str, re.Pattern[str], _Validator | None, str], ...] = (
    ("secret", SECRET, None, "[Zugangsdaten entfernt]"),
    ("pii_iban", _IBAN, _Validator(valid_iban, prefixes=True), "[IBAN entfernt]"),
    ("pii_payment_card", _CARD, _Validator(valid_card, prefixes=True), "[Kartennummer entfernt]"),
    ("pii_email", EMAIL, None, "[E-Mail-Adresse entfernt]"),
    ("pii_phone", PHONE, _Validator(valid_phone, prefixes=False), "[Telefonnummer entfernt]"),
)


def mask_personal_data(text: str, *, kinds: Iterable[str] | None = None) -> tuple[str, list[str]]:
    """Replace secrets, IBANs, card numbers, e-mail addresses and phone numbers.

    Returns the masked text and the reason codes of what was found (``pii_iban``, ...). Order
    numbers (``NE-123456``), prices, dates and parcel tracking numbers are left alone.
    """
    wanted = set(kinds) if kinds is not None else None
    found: list[str] = []
    for reason, pattern, valid, mask in _MASKS:
        if wanted is not None and reason not in wanted:
            continue
        masked = _mask(text, pattern, valid, mask)
        if masked != text:
            found.append(reason)
            text = masked
    return text, found


# --- tokens that must be verifiable in an answer --------------------------------------------

URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
# Voucher-like codes: upper-case letters and digits mixed, at least 5 characters (GRATIS100).
CODE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{3,}[A-Z0-9]\b")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
