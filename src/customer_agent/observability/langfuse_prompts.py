"""Prompts in Langfuse's prompt management (rebuild phase 4); git stays the source.

``make prompts-sync`` publishes every version under ``prompts/`` as a Langfuse chat prompt: name
``router``, ``answer``, …, label ``v<version>``, and ``production`` on the versions the service
uses. At start-up the service fetches those versions in the background and ties each model call
to its prompt version in the trace, but only while Langfuse holds exactly the text from git. The
service always renders from git: CI, the cassette and the tests never depend on Langfuse.
"""

import threading
from collections.abc import Iterable
from string import Template
from typing import Any, Literal, Protocol

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.model import ChatMessageDict, ChatPromptClient

from customer_agent.observability.logs import get_logger

log = get_logger("langfuse")

Outcome = Literal["created", "labelled", "unchanged"]

_linked: dict[str, ChatPromptClient] = {}  # ref → the Langfuse version with the same text


class PromptSource(Protocol):
    """A prompt version from git (``application.prompts.PromptTemplate``)."""

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> int: ...
    @property
    def description(self) -> str: ...
    @property
    def system(self) -> str: ...
    @property
    def user(self) -> str: ...
    @property
    def ref(self) -> str: ...


def langfuse_messages(prompt: PromptSource) -> list[ChatMessageDict]:
    """The prompt as Langfuse stores it: ``$name`` placeholders as ``{{name}}``, ``$$`` as ``$``.
    The system message has no placeholders; it is taken as written."""
    parts: list[str] = []
    end = 0
    for match in Template.pattern.finditer(prompt.user):
        parts.append(prompt.user[end : match.start()])
        name = match.group("named") or match.group("braced")
        parts.append("{{" + name + "}}" if name else "$")
        end = match.end()
    parts.append(prompt.user[end:])
    return [
        ChatMessageDict(role="system", content=prompt.system),
        ChatMessageDict(role="user", content="".join(parts)),
    ]


def _same_text(client: Any, prompt: PromptSource) -> bool:
    stored = [{"role": m["role"], "content": m["content"]} for m in client.prompt]
    return bool(stored == langfuse_messages(prompt))


def sync_prompt(client: Langfuse, prompt: PromptSource, *, production: bool) -> Outcome:
    """Publish one version unless Langfuse has it with the same text; move ``production``."""
    label = f"v{prompt.version}"
    labels = [label, "production"] if production else [label]
    try:
        current = client.get_prompt(prompt.name, label=label, type="chat", cache_ttl_seconds=0)
    except Exception:  # not published yet (the SDK raises a not-found error)
        current = None
    if current is not None and _same_text(current, prompt):
        if not production or "production" in current.labels:
            return "unchanged"
        client.update_prompt(name=prompt.name, version=current.version, new_labels=labels)
        return "labelled"
    client.create_prompt(
        name=prompt.name,
        type="chat",
        prompt=list(langfuse_messages(prompt)),
        labels=labels,
        config={"git": prompt.ref},
        commit_message=f"{prompt.ref} from git: {prompt.description.strip()[:200]}",
    )
    return "created"


def prefetch_prompts(client: Langfuse, prompts: Iterable[PromptSource]) -> threading.Thread:
    """Fetch the versions in use in the background: a turn never waits for Langfuse. A version
    whose text differs from git is not linked (and logged), so a trace never shows a prompt the
    model did not get."""
    wanted = list(prompts)

    def fetch() -> None:
        for prompt in wanted:
            try:
                found = client.get_prompt(
                    prompt.name, label=f"v{prompt.version}", type="chat", max_retries=1
                )
            except Exception as exc:  # not synced, or Langfuse unreachable
                log.warning("prompt %s not in Langfuse: %s", prompt.ref, type(exc).__name__)
                continue
            if isinstance(found, ChatPromptClient) and _same_text(found, prompt):
                _linked[prompt.ref] = found
            else:
                log.warning("prompt %s differs in Langfuse; run make prompts-sync", prompt.ref)

    thread = threading.Thread(target=fetch, name="langfuse-prompts", daemon=True)
    thread.start()
    return thread


def prompt_config(ref: str) -> RunnableConfig:
    """For the prompt's rendering run: ties the next model call to the prompt version in
    Langfuse; empty while the version is unknown (Langfuse off, not fetched, or different)."""
    linked = _linked.get(ref)
    return {"metadata": {"langfuse_prompt": linked}} if linked is not None else {}


def forget_prompts() -> None:
    """On shutdown, with the client."""
    _linked.clear()
