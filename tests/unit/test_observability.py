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


# --- traces and metrics ----------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

from customer_agent.config import Settings  # noqa: E402
from customer_agent.domain.conversation import Conversation, Message, Role  # noqa: E402
from customer_agent.main import create_app  # noqa: E402
from customer_agent.observability import configure_tracing, step, tracer  # noqa: E402
from tests.fakes import ScriptedRouterLLM, make_orchestrator  # noqa: E402

_EXPORTER = InMemorySpanExporter()
configure_tracing("test", "0", None).add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_step_records_a_span_and_a_duration_sample(spans: InMemorySpanExporter) -> None:
    before = sample("ca_step_duration_seconds_count", step="unit", outcome="ok")

    with step("unit", documents=2):
        pass

    [span] = spans.get_finished_spans()
    assert span.name == "agent.unit" and span.attributes == {"documents": 2}
    assert sample("ca_step_duration_seconds_count", step="unit", outcome="ok") == before + 1


def test_failed_step_marks_span_and_metric(spans: InMemorySpanExporter) -> None:
    before = sample("ca_step_duration_seconds_count", step="unit", outcome="error")

    with pytest.raises(ValueError), step("unit"):
        raise ValueError("boom")

    [span] = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert sample("ca_step_duration_seconds_count", step="unit", outcome="error") == before + 1


async def test_one_turn_is_one_trace_with_a_span_per_step(spans: InMemorySpanExporter) -> None:
    routing = '{"intent": "RETURN_POLICY", "confidence": 0.9, "standalone_query": "Kosten?"}'
    agent = make_orchestrator(ScriptedRouterLLM(routing))

    with tracer.start_as_current_span("request"):
        await agent.respond(
            Conversation(messages=[Message(role=Role.USER, content="Was kostet das?")])
        )

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert {"agent.understand", "agent.route", "agent.output_guardrail"} <= set(names)
    assert len({s.context.trace_id for s in finished}) == 1
    by_name = {s.name: s for s in finished}  # the router call is a step inside the agent
    route_parent = by_name["agent.route"].parent
    assert route_parent is not None
    assert route_parent.span_id == by_name["agent.understand"].context.span_id


def test_request_span_continues_an_incoming_trace_and_logs_carry_it(
    spans: InMemorySpanExporter,
) -> None:
    app = create_app(Settings(app_env="test"), chat=object())  # type: ignore[arg-type]
    buffer = io.StringIO()
    configure_logging("INFO", "json", stream=buffer)
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    with TestClient(app) as client:
        client.get("/health", headers={"traceparent": f"00-{trace_id}-00f067aa0ba902b7-01"})

    configure_logging("INFO", "console")
    [server] = [s for s in spans.get_finished_spans() if s.name == "GET /health"]
    assert format(server.context.trace_id, "032x") == trace_id
    assert server.attributes is not None and server.attributes["http.route"] == "/health"
    assert server.attributes["request_id"] == lines(buffer)[-1]["request_id"]
    assert lines(buffer)[-1]["trace_id"] == trace_id


def test_http_metrics_use_route_templates_and_are_exposed() -> None:
    app = create_app(Settings(app_env="test"), chat=object())  # type: ignore[arg-type]
    before = sample("ca_http_requests_total", method="GET", route="unmatched", status="404")

    with TestClient(app) as client:
        client.get("/does-not-exist/12345")
        text = client.get("/metrics").text

    assert sample("ca_http_requests_total", method="GET", route="unmatched", status="404") == (
        before + 1
    )
    assert "ca_http_request_duration_seconds_bucket" in text
