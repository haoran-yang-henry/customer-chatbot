"""Personal data and secrets in free text: detection with checksums, masking with placeholders.

Shared by the input and output guardrails (ADR-0011) and the log redaction (Day 7): one
definition of what counts as personal data, so logs never keep what the chat already masks.
Pure functions, no I/O.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

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
