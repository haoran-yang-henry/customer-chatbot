"""Detection patterns shared by the input, evidence and output guardrails (ADR-0011).

Deterministic and explainable: every injection hit has a signal id that appears in logs and
evaluation reports. Patterns cover German and English, because attacks against a German
assistant are often written in English. They are a first line, not the only one: paraphrases
they miss are left to the LLM classifier and to the layers behind it.
"""

import re
import unicodedata
from collections.abc import Iterable
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
        _p(r"\b(do anything now|god ?mode|unrestricted mode|uneingeschränkte[nr]? modus)\b"),
    ),
    # Words with an ordinary meaning in an electronics shop ("Smartphone jailbreaken", Android's
    # "Entwicklermodus"): alone they do not block, together with an override or "ohne Regeln"
    # they do (demo test 2026-10-15).
    Signal(
        "jailbreak_term",
        0.5,
        _p(r"\b(developer mode|dev mode|entwicklermodus|entwickleroptionen|jailbreak\w*)\b"),
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


# --- tokens that must be verifiable in an answer --------------------------------------------

URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
# Voucher-like codes: upper-case letters and digits mixed, at least 5 characters (GRATIS100).
CODE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{3,}[A-Z0-9]\b")
