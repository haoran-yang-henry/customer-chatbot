"""Observability: structured logs (``logs``), traces and metrics (Day 7, see the architecture
document §7). Everything other modules need is importable from here."""

from customer_agent.observability.logs import (
    LogFormat,
    configure_logging,
    elapsed_ms,
    get_logger,
    request_id_var,
    session_id_var,
)

__all__ = [
    "LogFormat",
    "configure_logging",
    "elapsed_ms",
    "get_logger",
    "request_id_var",
    "session_id_var",
]
