"""Langfuse (rebuild plan phases 1 and 3): chat turns, agent runs and model calls as traces.

The client joins the project's TracerProvider, so logs, Jaeger and Langfuse share one trace id;
the request id travels as trace metadata. LangChain and LangGraph runs (agents, middleware
hooks, graph nodes, model calls, the retriever, tools) reach Langfuse through its callback
handler, one per turn (``langchain_callbacks``). Only spans of the Langfuse SDK and the agent
steps of this package are exported: FastAPI and database spans stay in Jaeger. Every payload
passes the log masking before it leaves the process (NFR-12). Langfuse is not part of
readiness: when it is unreachable, spans are dropped in the background and turns are not
slowed down.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, LangfuseSpan, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.span_filter import is_default_export_span
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from customer_agent.domain.pii import mask_personal_data
from customer_agent.observability.logs import request_id_var

AGENT_SCOPE = "customer_agent"  # the tracer behind observability.step

_client: Langfuse | None = None
_public_key: str | None = None  # identifies _client to the LangChain callback handler


def mask(*, data: Any, **_: Any) -> Any:
    """Langfuse's mask hook: every string in a payload is masked like a log line."""
    if isinstance(data, str):
        return mask_personal_data(data)[0]
    if isinstance(data, dict):
        return {key: mask(data=value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [mask(data=value) for value in data]
    return data


def exported(span: ReadableSpan) -> bool:
    """Langfuse's own spans, LLM instrumentation and the agent steps; nothing else."""
    scope = span.instrumentation_scope
    return is_default_export_span(span) or (scope is not None and scope.name == AGENT_SCOPE)


def configure_langfuse(
    *,
    public_key: str,
    secret_key: str,
    base_url: str | None,
    environment: str,
    release: str,
    tracer_provider: TracerProvider,
    timeout: int = 5,
) -> Langfuse:
    """The tracer provider is passed explicitly: on its own, Langfuse would install a global one
    if it started first, and configure_tracing would then skip the Jaeger exporter."""
    global _client, _public_key  # one client per process, like the tracer provider
    _public_key = public_key
    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        release=release,
        timeout=timeout,
        mask=mask,
        should_export_span=exported,
        tracer_provider=tracer_provider,
    )
    return _client


def langfuse_client() -> Langfuse | None:
    """The configured client, for the evaluation's experiments; ``None`` when off."""
    return _client


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """For one turn's LangChain config: a new handler per turn, because a shared one's trace
    bookkeeping is unreliable under concurrent turns. Empty when Langfuse is off."""
    if _client is None:
        return []
    # Named by key: without it the handler looks up "the" client and records nothing when a
    # process holds more than one.
    return [CallbackHandler(public_key=_public_key)]


def shutdown_langfuse() -> None:
    """Send what is queued (bounded by the client timeout) and forget the client."""
    global _client
    if _client is not None:
        _client.shutdown()
        _client = None


def score_session(
    session_id: str, *, helpful: bool, comment: str | None, asked_in: str | None
) -> None:
    """The customer's rating of a conversation, on the Langfuse session (= our session id);
    ``asked_in``: the trace of the turn that asked. Nothing when Langfuse is off. ``comment``
    must already be masked."""
    if _client is not None:
        _client.create_score(
            session_id=session_id,
            name="user_feedback",
            value=1 if helpful else 0,
            data_type="BOOLEAN",
            comment=comment,
            metadata={"asked_in_trace": asked_in} if asked_in else None,
        )


class TurnTrace:
    def __init__(self, span: LangfuseSpan | None) -> None:
        self._span = span

    def record(self, *, user_text: str, answer: str, metadata: dict[str, str]) -> None:
        """``user_text`` is the message as the input guardrail left it, never the raw one."""
        if self._span is not None:
            self._span.update(input=user_text, output=answer, metadata=metadata)


@contextmanager
def trace_turn(*, session_id: str, customer_id: str | None) -> Iterator[TurnTrace]:
    """One chat turn: the root of its agent steps and model calls in Langfuse; a no-op when
    Langfuse is not configured."""
    if _client is None:
        yield TurnTrace(None)
        return
    with (
        _client.start_as_current_observation(as_type="span", name="chat-turn") as span,
        propagate_attributes(
            session_id=session_id,
            user_id=customer_id,
            trace_name="chat-turn",
            metadata={"request_id": request_id_var.get()},
        ),
    ):
        yield TurnTrace(span)
