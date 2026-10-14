"""Prometheus metrics (Day 7), exposed at ``GET /metrics``.

Names carry the ``ca_`` prefix; label values come from small, fixed sets (routes, steps, models,
reason codes), never from user input, so the number of time series stays bounded.
"""

from prometheus_client import Counter, Gauge, Histogram

from customer_agent.domain.guardrails import GuardrailDecision

# Latency buckets from 5 ms to 30 s: system steps live at the low end, LLM calls at the high end.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

HTTP_REQUESTS = Counter("ca_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_DURATION = Histogram(
    "ca_http_request_duration_seconds",
    "HTTP request duration, including a streamed body",
    ["method", "route"],
    buckets=_BUCKETS,
)
STEP_DURATION = Histogram(
    "ca_step_duration_seconds",
    "Duration of one agent step (route, classify, retrieve, llm, output_guardrail, tools)",
    ["step", "outcome"],
    buckets=_BUCKETS,
)
LLM_TOKENS = Counter("ca_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_FIRST_TOKEN = Histogram(
    "ca_llm_time_to_first_token_seconds",
    "Time from request to the first streamed token (NFR-10)",
    ["model"],
    buckets=_BUCKETS,
)
GUARDRAIL_DECISIONS = Counter(
    "ca_guardrail_decisions_total", "Guardrail decisions by reason", ["stage", "action", "reason"]
)
AGENT_REPLIES = Counter("ca_agent_replies_total", "Agent replies", ["behavior", "intent"])
PROVIDER_FAILURES = Counter(
    "ca_provider_failures_total", "Failed calls to an external model provider", ["dependency"]
)
CIRCUIT_OPEN = Gauge(
    "ca_circuit_open", "1 while the circuit breaker of a dependency is open", ["dependency"]
)
RATE_LIMITED = Counter("ca_rate_limited_total", "Requests rejected with 429", ["scope"])


def record_tokens(model: str, prompt: int | None, completion: int | None) -> None:
    if prompt:
        LLM_TOKENS.labels(model=model, kind="prompt").inc(prompt)
    if completion:
        LLM_TOKENS.labels(model=model, kind="completion").inc(completion)


def token_totals() -> dict[str, float]:
    """Tokens counted so far in this process, by kind (all models)."""
    totals = {"prompt": 0.0, "completion": 0.0}
    for metric in LLM_TOKENS.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                totals[sample.labels["kind"]] += sample.value
    return totals


def record_guardrail(decision: GuardrailDecision) -> None:
    """One sample per reason; a decision without reasons counts as reason ``none``."""
    for reason in decision.reasons or ["none"]:
        GUARDRAIL_DECISIONS.labels(
            stage=decision.stage, action=decision.action, reason=reason
        ).inc()
