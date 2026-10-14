"""Structured logs (Day 7): context, fields, redaction (NFR-12)."""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from customer_agent.observability import (
    configure_logging,
    get_logger,
    request_id_var,
    session_id_var,
)


@pytest.fixture
def json_log() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    configure_logging("INFO", "json", stream=buffer)
    request_token = request_id_var.set("req-1")
    session_token = session_id_var.set("s-42")
    yield buffer
    request_id_var.reset(request_token)
    session_id_var.reset(session_token)
    configure_logging("INFO", "console")


def lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_json_lines_carry_context_and_message_fields(json_log: io.StringIO) -> None:
    get_logger("answer").info("retrieved=%s took=%dms", "policy-returns-v5", 12)

    [line] = lines(json_log)
    assert line["event"] == "retrieved=policy-returns-v5 took=12ms"
    assert (line["level"], line["component"]) == ("info", "answer")
    assert (line["request_id"], line["session_id"]) == ("req-1", "s-42")
    assert (line["retrieved"], line["took"]) == ("policy-returns-v5", "12ms")


def test_personal_data_is_masked_in_messages_and_exceptions(json_log: io.StringIO) -> None:
    log = get_logger("chat")
    log.warning("Rückruf an max@web.de, IBAN DE89 3704 0044 0532 0130 00")
    try:
        raise ValueError("Karte 4111 1111 1111 1111 abgelehnt")
    except ValueError:
        log.exception("turn failed")

    text = json_log.getvalue()
    assert "max@web.de" not in text and "DE89" not in text and "4111 1111" not in text
    assert "[E-Mail-Adresse entfernt]" in text and "[Kartennummer entfernt]" in text


def test_identifiers_are_never_redacted(json_log: io.StringIO) -> None:
    token = request_id_var.set("4111111111111111")  # a valid card number as request id

    get_logger("http").info("GET /health")

    request_id_var.reset(token)
    assert lines(json_log)[0]["request_id"] == "4111111111111111"


def test_configuration_is_idempotent() -> None:
    configure_logging("INFO", "json")
    configure_logging("DEBUG", "console")

    logger = logging.getLogger("customer_agent")
    assert len(logger.handlers) == 1 and logger.level == logging.DEBUG
    configure_logging("INFO", "console")
