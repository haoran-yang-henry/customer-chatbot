"""German formats for customer-facing text (LR-04): ``6. Oktober 2026``, ``1.099,00 €``."""

from datetime import date
from decimal import Decimal

_MONTHS = (
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
)  # fmt: skip


def format_date_de(day: date) -> str:
    return f"{day.day}. {_MONTHS[day.month - 1]} {day.year}"


def format_eur(amount: Decimal) -> str:
    whole, _, cents = f"{amount.quantize(Decimal('0.01')):,.2f}".partition(".")
    return f"{whole.replace(',', '.')},{cents} €"
