"""Is this text German? A function-word count, deterministic and dependency-free (LR-03).

Function words are frequent, short and language-specific, so a handful of them decides the
language of a customer-service answer reliably. Words that exist in both languages ("in",
"an", "will", "was", "die") are left out of both lists.
"""

import re

_GERMAN = frozenset((
    "der", "das", "und", "ist", "nicht", "sie", "ihre", "ihr", "ihnen", "ein", "eine", "einen",
    "einem", "zu", "mit", "für", "auf", "bei", "von", "wir", "können", "kann", "werden", "wird",
    "oder", "wenn", "dass", "auch", "nach", "bis", "im", "den", "dem", "des", "es", "sich",
    "als", "noch", "nur", "wie", "uns", "unser", "unsere", "bitte", "gern", "gerne", "sind",
    "hat", "haben", "ab", "sowie", "über", "unter", "wurde", "innerhalb",
))  # fmt: skip
_ENGLISH = frozenset((
    "the", "and", "is", "you", "your", "to", "of", "for", "with", "can", "are", "this", "that",
    "it", "be", "we", "our", "please", "have", "has", "not", "on", "if", "or", "as", "at", "by",
    "from", "do", "does", "would", "could", "should", "there", "their", "which", "within",
))  # fmt: skip
_WORD = re.compile(r"[a-zäöüß]+")
MIN_MARKERS = 3  # fewer function words than this: too short to judge


def looks_german(text: str) -> bool | None:
    """``True``/``False`` when the text has enough function words to decide, else ``None``."""
    words = _WORD.findall(text.casefold())
    german = sum(word in _GERMAN for word in words)
    english = sum(word in _ENGLISH for word in words)
    if german + english < MIN_MARKERS:
        return None
    return german >= english
