"""Traces (Day 7): one trace per request, one span per agent step.

The SDK provider is installed even without an exporter, so log lines always carry a trace id;
spans leave the process only when ``CA_OTEL_ENDPOINT`` points to an OTLP/HTTP receiver (Jaeger,
an OpenTelemetry Collector).
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from customer_agent.observability.metrics import STEP_DURATION

tracer = trace.get_tracer("customer_agent")


def configure_tracing(service: str, version: str, endpoint: str | None) -> TracerProvider:
    """Idempotent: the global provider can be set once per process."""
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    provider = TracerProvider(
        resource=Resource.create({"service.name": service, "service.version": version})
    )
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def current_trace_id() -> str | None:
    """The current trace's id as Jaeger and Langfuse show it (32 hex digits); ``None`` outside
    a trace."""
    context = trace.get_current_span().get_span_context()
    return trace.format_trace_id(context.trace_id) if context.is_valid else None


@contextmanager
def step(name: str, **attributes: str | int | float | bool) -> Iterator[trace.Span]:
    """One agent step: a child span of the current one plus a duration sample.

    Use it around awaited calls, not around ``yield`` in an async generator: the span context
    must be detached in the same context it was attached in.
    """
    start = time.perf_counter()
    outcome = "ok"
    with tracer.start_as_current_span(f"agent.{name}", attributes=attributes) as span:
        try:
            yield span
        except Exception:
            outcome = "error"  # the span records the exception and sets the error status
            raise
        finally:
            STEP_DURATION.labels(step=name, outcome=outcome).observe(time.perf_counter() - start)
