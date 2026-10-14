"""Structured logs (Day 3 foundation, Day 7 structlog).

Call sites keep using stdlib logging with ``key=value`` messages (``log.info("took=%dms", ms)``);
structlog renders every record either as one JSON object per line (``CA_LOG_FORMAT=json``,
production) or as a readable console line (local). Each record carries the request id, the
session id and the trace/span ids of the current span, and personal data is masked before
anything is written (NFR-12): the input guardrail masks customer messages, this masks whatever
else might slip into a log line, such as an exception message.
"""

import logging
import re
import sys
import time
from contextvars import ContextVar
from typing import Literal, TextIO

import structlog
from opentelemetry import trace
from structlog.typing import EventDict, Processor, WrappedLogger

from customer_agent.domain.pii import mask_personal_data

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")

LogFormat = Literal["console", "json"]
_PREFIX = "customer_agent."
_KEY_VALUE = re.compile(r"\b(\w+)=([^\s,;]+)")
# Third-party loggers rendered the same way (uvicorn's startup and error messages).
_ALSO_CONFIGURED = ("uvicorn", "uvicorn.error")


def _context(logger: WrappedLogger, method: str, event: EventDict) -> EventDict:
    event["component"] = event.pop("logger", "").removeprefix(_PREFIX)
    event["request_id"] = request_id_var.get()
    if (session := session_id_var.get()) != "-":
        event["session_id"] = session
    span = trace.get_current_span().get_span_context()
    if span.is_valid:
        event["trace_id"] = format(span.trace_id, "032x")
        event["span_id"] = format(span.span_id, "016x")
    return event


def _fields(logger: WrappedLogger, method: str, event: EventDict) -> EventDict:
    """``key=value`` pairs of the message become queryable fields; the message stays intact."""
    for key, value in _KEY_VALUE.findall(str(event.get("event", ""))):
        event.setdefault(key, value)
    return event


# Identifiers are left alone: a 32-digit hex trace id can contain a Luhn-valid digit run.
_NOT_REDACTED = frozenset(
    {"timestamp", "level", "component", "request_id", "session_id", "trace_id", "span_id"}
)


def _redact(logger: WrappedLogger, method: str, event: EventDict) -> EventDict:
    for key, value in event.items():
        if isinstance(value, str) and key not in _NOT_REDACTED:
            event[key] = mask_personal_data(value)[0]
    return event


def _processors(fmt: LogFormat) -> tuple[list[Processor], Processor]:
    shared: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _context,
        structlog.processors.format_exc_info,
    ]
    if fmt == "json":
        return [*shared, _fields, _redact], structlog.processors.JSONRenderer()
    return [*shared, _redact], structlog.dev.ConsoleRenderer(colors=False)


def configure_logging(
    level: str = "INFO", fmt: LogFormat = "console", stream: TextIO | None = None
) -> None:
    """Idempotent: safe to call from every entry point. ``stream`` is for tests."""
    pre_chain, renderer = _processors(fmt)
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    for name in ("customer_agent", *_ALSO_CONFIGURED):
        logger = logging.getLogger(name)
        logger.handlers[:] = [handler]
        logger.propagate = False
    logging.getLogger("customer_agent").setLevel(level.upper())


def get_logger(component: str) -> logging.Logger:
    """Short component names (``chat``, ``retrieval``, ``llm``) keep log lines readable."""
    return logging.getLogger(f"{_PREFIX}{component}")


def elapsed_ms(start: float) -> int:
    """Milliseconds since ``start = time.perf_counter()``."""
    return round((time.perf_counter() - start) * 1000)
