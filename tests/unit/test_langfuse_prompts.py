"""Prompts in Langfuse (rebuild phase 4): published from git, linked to the model calls."""

from typing import Any

import pytest
from langfuse.api import Prompt_Chat
from langfuse.model import ChatPromptClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from customer_agent.application.guardrails.classifier import PromptAttackClassifier
from customer_agent.application.prompts import PromptTemplate
from customer_agent.domain.conversation import Conversation, Message, Role
from customer_agent.infrastructure.prompts import PROMPTS_DIR, load_prompt
from customer_agent.observability import langfuse_prompts
from customer_agent.observability.langfuse_prompts import (
    langfuse_messages,
    prefetch_prompts,
    prompt_config,
    sync_prompt,
)
from customer_agent.observability.langfuse_tracing import (
    configure_langfuse,
    shutdown_langfuse,
    trace_turn,
)
from tests.fakes import FakeLLM, ScriptedRouterLLM, make_orchestrator

SHIPPED = sorted(
    (path.parent.name, int(path.stem.removeprefix("v"))) for path in PROMPTS_DIR.glob("*/v*.yaml")
)


def published(prompt: PromptTemplate, *, version: int = 7, labels: list[str] | None = None) -> Any:
    """The prompt as Langfuse returns it after a sync (Langfuse numbers its versions itself)."""
    stored = {
        "name": prompt.name,
        "version": version,
        "prompt": langfuse_messages(prompt),
        "config": {"git": prompt.ref},
        "labels": labels or [f"v{prompt.version}"],
        "tags": [],
        "type": "chat",
    }
    return ChatPromptClient(Prompt_Chat.model_validate(stored))


@pytest.mark.parametrize(("name", "version"), SHIPPED)
def test_langfuse_renders_every_shipped_prompt_exactly_like_git(name: str, version: int) -> None:
    prompt = load_prompt(name, version)
    values = {key: f"<{key}>" for key in prompt.chat_prompt.input_variables}

    compiled = published(prompt).compile(**values)

    ours = prompt.render_messages(**values)
    assert [m["content"] for m in compiled] == [m.text for m in ours]


def test_placeholders_become_mustache_and_escaped_dollars_stay_dollars() -> None:
    prompt = PromptTemplate(
        name="t", version=1, description="d", system="System $x {json}", user="A $a ${b} $$5 {c}"
    )

    assert langfuse_messages(prompt) == [
        {"role": "system", "content": "System $x {json}"},
        {"role": "user", "content": "A {{a}} {{b}} $5 {c}"},
    ]


class FakeClient:
    """The two calls the sync and the fetch make, against an in-memory prompt store."""

    def __init__(self, stored: dict[tuple[str, str], Any] | None = None) -> None:
        self.stored = stored or {}
        self.created: list[dict[str, Any]] = []
        self.relabelled: list[tuple[str, int, list[str]]] = []

    def get_prompt(self, name: str, *, label: str, **_: Any) -> Any:
        if (name, label) not in self.stored:
            raise LookupError("prompt not found")
        return self.stored[(name, label)]

    def create_prompt(self, **prompt: Any) -> None:
        self.created.append(prompt)

    def update_prompt(self, *, name: str, version: int, new_labels: list[str]) -> None:
        self.relabelled.append((name, version, new_labels))


ROUTER = load_prompt("router", 5)


def test_a_new_version_is_published_with_its_labels() -> None:
    client = FakeClient()

    assert sync_prompt(client, ROUTER, production=True) == "created"  # type: ignore[arg-type]

    [created] = client.created
    assert (created["name"], created["type"], created["labels"]) == (
        "router",
        "chat",
        ["v5", "production"],
    )
    assert created["prompt"] == langfuse_messages(ROUTER)
    assert created["config"] == {"git": "router/v5"}


def test_a_version_langfuse_has_is_left_alone_and_production_moves_to_it() -> None:
    same = FakeClient({("router", "v5"): published(ROUTER, labels=["v5", "production"])})
    unlabelled = FakeClient({("router", "v5"): published(ROUTER, version=3)})

    assert sync_prompt(same, ROUTER, production=True) == "unchanged"  # type: ignore[arg-type]
    assert sync_prompt(unlabelled, ROUTER, production=True) == "labelled"  # type: ignore[arg-type]
    assert same.created == [] and unlabelled.created == []
    assert unlabelled.relabelled == [("router", 3, ["v5", "production"])]


def test_a_version_edited_in_langfuse_is_published_again_from_git() -> None:
    edited = published(load_prompt("router", 4))  # other text under the label v5
    client = FakeClient({("router", "v5"): edited})

    assert sync_prompt(client, ROUTER, production=False) == "created"  # type: ignore[arg-type]
    assert client.created[0]["labels"] == ["v5"]


@pytest.fixture
def no_links() -> Any:
    yield
    langfuse_prompts.forget_prompts()


def test_only_a_version_with_the_text_from_git_is_linked(no_links: None) -> None:
    answer, guard = load_prompt("answer", 4), load_prompt("guard", 1)
    client = FakeClient(
        {
            ("router", "v5"): published(ROUTER),
            ("answer", "v4"): published(load_prompt("answer", 3)),  # differs from git
        }
    )

    prefetch_prompts(client, [ROUTER, answer, guard]).join()  # type: ignore[arg-type]

    assert prompt_config("router/v5")["metadata"]["langfuse_prompt"].version == 7
    assert prompt_config("answer/v4") == {}  # differs
    assert prompt_config("guard/v1") == {}  # not synced


async def test_each_model_call_of_a_turn_is_tied_to_its_prompt_version(no_links: None) -> None:
    """The classifier renders beside the router: each generation must get its own prompt."""
    finished = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))
    configure_langfuse(
        public_key="pk-lf-prompts",  # its own key: Langfuse keeps one client per key
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        environment="test",
        release="test",
        tracer_provider=provider,
        timeout=1,
    )
    guard, answer = load_prompt("guard", 1), load_prompt("answer", 4)
    langfuse_prompts._linked.update(
        {p.ref: published(p, version=v) for p, v in ((ROUTER, 11), (guard, 12), (answer, 13))}
    )
    routing = '{"intent": "RETURN_POLICY", "confidence": 0.9, "standalone_query": "Kosten?"}'
    classifier = PromptAttackClassifier(
        FakeLLM('{"attack_probability": 0.0, "category": "none"}'), guard, threshold=0.7
    )
    try:
        with trace_turn(session_id="s-1", customer_id=None):
            await make_orchestrator(ScriptedRouterLLM(routing), classifier=classifier).respond(
                Conversation(messages=[Message(role=Role.USER, content="Was kostet das?")])
            )
    finally:
        shutdown_langfuse()

    linked = sorted(
        ((span.attributes or {}).get("langfuse.observation.prompt.version"), span.name)
        for span in finished.get_finished_spans()
        if (span.attributes or {}).get("langfuse.observation.type") == "generation"
    )
    # Router and answer use the same fake model (one name); the classifier names its call.
    assert linked == [(11, "ScriptedRouterLLM"), (12, "classifier"), (13, "ScriptedRouterLLM")]


def test_without_langfuse_no_prompt_is_linked() -> None:
    assert prompt_config("router/v5") == {}
