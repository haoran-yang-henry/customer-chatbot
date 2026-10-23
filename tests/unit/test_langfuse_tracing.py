"""Langfuse tracing (rebuild phase 1): masking, span filter, when it is on, and an outage."""

import time
from typing import Literal

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from customer_agent.config import Settings
from customer_agent.domain.conversation import Conversation, Message, Role
from customer_agent.observability.langfuse_tracing import (
    configure_langfuse,
    exported,
    mask,
    shutdown_langfuse,
    trace_turn,
)
from tests.fakes import ScriptedRouterLLM, make_orchestrator

IBAN = "DE89 3704 0044 0532 0130 00"


def test_mask_masks_every_string_in_a_payload() -> None:
    payload = {
        "messages": [{"role": "user", "content": f"Meine IBAN ist {IBAN}"}],
        "pair": ("Hallo", IBAN),
        "tokens": 3,
        "streamed": True,
    }

    masked = mask(data=payload)

    assert IBAN not in str(masked)
    assert masked["messages"][0]["content"] == "Meine IBAN ist [IBAN entfernt]"
    assert masked["tokens"] == 3
    assert masked["streamed"] is True


def test_only_langfuse_and_agent_step_spans_are_exported() -> None:
    finished = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))

    with provider.get_tracer("customer_agent").start_as_current_span("agent.route"):
        pass
    with provider.get_tracer("fastapi").start_as_current_span("GET /health"):
        pass

    assert {span.name: exported(span) for span in finished.get_finished_spans()} == {
        "agent.route": True,
        "GET /health": False,
    }


@pytest.mark.parametrize(
    ("app_env", "secret_key", "enabled"),
    [
        ("local", "sk-lf-1", True),
        ("prod", "sk-lf-1", True),
        ("test", "sk-lf-1", False),
        ("ci", "sk-lf-1", False),
        ("local", "", False),
    ],
)
def test_langfuse_needs_both_keys_and_is_never_on_in_tests_or_ci(
    monkeypatch: pytest.MonkeyPatch,
    app_env: Literal["local", "test", "ci", "prod"],
    secret_key: str,
    enabled: bool,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)

    assert Settings(_env_file=None, app_env=app_env).langfuse_enabled is enabled


def test_a_turn_trace_without_langfuse_does_nothing() -> None:
    with trace_turn(session_id="s-1", customer_id=None) as turn:
        turn.record(user_text="Hallo", answer="Guten Tag!", metadata={"behavior": "answer"})


def test_an_unreachable_langfuse_does_not_slow_a_turn() -> None:
    configure_langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",  # nothing listens here
        environment="test",
        release="test",
        tracer_provider=TracerProvider(),
        timeout=1,
    )
    try:
        start = time.perf_counter()
        with trace_turn(session_id="s-1", customer_id="C-1001") as turn:
            turn.record(user_text="Hallo", answer="Guten Tag!", metadata={"behavior": "answer"})
        assert time.perf_counter() - start < 0.2
    finally:
        shutdown_langfuse()


async def test_a_turn_reaches_langfuse_with_its_runs_and_without_the_raw_message() -> None:
    """Invariant I3 for traces: the understand agent's input is the raw message; the mask hook
    keeps it out of everything Langfuse would receive."""
    finished = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))
    configure_langfuse(
        # Its own key: Langfuse keeps one client per key, also after another test shut it down.
        public_key="pk-lf-i3",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",  # nothing listens here; spans are captured above
        environment="test",
        release="test",
        tracer_provider=provider,
        timeout=1,
    )
    routing = '{"intent": "RETURN_POLICY", "confidence": 0.9, "standalone_query": "Erstattung?"}'
    message = f"Erstattung bitte an {IBAN}, wie lange dauert das?"
    try:
        with trace_turn(session_id="s-1", customer_id=None):
            await make_orchestrator(ScriptedRouterLLM(routing)).respond(
                Conversation(messages=[Message(role=Role.USER, content=message)])
            )
    finally:
        shutdown_langfuse()

    spans = finished.get_finished_spans()
    payload = " ".join(str(dict(span.attributes or {})) for span in spans)
    names = {span.name for span in spans}
    assert {"chat-turn", "understand", "turn", "answer"} <= names
    assert len({span.context.trace_id for span in spans}) == 1
    assert "[IBAN entfernt]" in payload
    assert IBAN not in payload and IBAN.replace(" ", "") not in payload
