"""Logging foundation (Day 3). Full tracing and metrics follow on Day 7.

Every log line carries the current request id, set once per request by the API middleware and
propagated through async code by a ``ContextVar``. Messages use ``key=value`` pairs so they stay
greppable now and map one-to-one to structured fields later.
"""

import logging
import time
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_FORMAT = "%(asctime)s %(levelname)-5s %(component)-10s [%(request_id)s] %(message)s"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.component = record.name.removeprefix("customer_agent.")
        return True


def configure_logging(level: str = "INFO") -> None:
    """Idempotent: safe to call from every entry point."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(_RequestIdFilter())
    logger = logging.getLogger("customer_agent")
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(component: str) -> logging.Logger:
    """Short component names (``chat``, ``retrieval``, ``llm``) keep log lines readable."""
    return logging.getLogger(f"customer_agent.{component}")


def elapsed_ms(start: float) -> int:
    """Milliseconds since ``start = time.perf_counter()``."""
    return round((time.perf_counter() - start) * 1000)
