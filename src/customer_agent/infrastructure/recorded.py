"""Chat models that answer from recorded responses, for evaluation runs and the CI gate.

``CachedChatModel`` keeps answers on disk between evaluation runs (``CA_LLM_CACHE_DIR``);
``ReplayChatModel`` answers from the smoke cassette (ADR-0013, ``cassette.py``). Neither is ever
used in production. Both record the complete answer and give it back in one piece; a streamed
call gets it as a single chunk, with the recorded usage.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    UsageMetadata,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

from customer_agent.infrastructure.cassette import (
    RERECORD,
    Cassette,
    ChatMessage,
    LLMResult,
    ReplayMissError,
    chat_key,
)
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.models import AsyncOnlyChatModel, model_label
from customer_agent.observability import record_tokens

_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}


def as_chat_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """The storage shape of a request (cache and cassette keys predate LangChain)."""
    return [ChatMessage(role=_ROLES[type(m)], content=m.text) for m in messages]  # type: ignore[arg-type]


class RecordedChatModel(AsyncOnlyChatModel):
    def _answer(self, recorded: LLMResult, model: str) -> ChatResult:
        """The recorded answer; its usage counts, so a run's token totals are what it would
        cost."""
        record_tokens(model, recorded.prompt_tokens, recorded.completion_tokens)
        usage = None
        if recorded.prompt_tokens is not None and recorded.completion_tokens is not None:
            usage = UsageMetadata(
                input_tokens=recorded.prompt_tokens,
                output_tokens=recorded.completion_tokens,
                total_tokens=recorded.prompt_tokens + recorded.completion_tokens,
            )
        message = AIMessage(content=recorded.text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _recording(result: ChatResult, model: str) -> LLMResult | None:
        """What to record of a provider's answer; ``None`` for an empty generation, which is a
        failure, not an answer."""
        message = result.generations[0].message
        if not message.text.strip():
            return None
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        return LLMResult(
            text=message.text,
            model=model,
            prompt_tokens=usage["input_tokens"] if usage else None,
            completion_tokens=usage["output_tokens"] if usage else None,
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.text,
                usage_metadata=message.usage_metadata if isinstance(message, AIMessage) else None,
            )
        )


class CachedChatModel(RecordedChatModel):
    """On-disk answers for evaluation runs (``CA_LLM_CACHE_DIR``): identical requests are
    answered from disk. Never in production — answers would not reflect knowledge or prompt
    changes outside the request itself. ``namespace`` must identify everything besides the
    request that changes the output (endpoint, model, reasoning settings). Keys and files are
    those of the Day 3 cache, so existing caches stay valid."""

    inner: BaseChatModel
    directory: Path
    namespace: str

    @property
    def _llm_type(self) -> str:
        return "cached"

    async def aclose(self) -> None:
        await aclose_all(self.inner)

    def _path(self, messages: Sequence[BaseMessage], kwargs: dict[str, Any]) -> Path:
        key = json.dumps(
            [
                self.namespace,
                [m.model_dump() for m in as_chat_messages(messages)],
                kwargs.get("max_tokens"),
                kwargs.get("temperature"),
            ],
            ensure_ascii=False,
        )
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        path = self._path(messages, kwargs)
        # File I/O blocks; run it in the default thread pool so concurrent turns keep running.
        if await asyncio.to_thread(path.exists):
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            return self._answer(LLMResult.model_validate_json(text), model_label(self.inner))
        result = await self.inner._agenerate(messages, stop, run_manager, **kwargs)
        if (entry := self._recording(result, model_label(self.inner))) is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, entry.model_dump_json(), encoding="utf-8")
        return result


class ReplayChatModel(RecordedChatModel):
    """Answers from the cassette (ADR-0013); in record mode, asks ``inner`` for misses. Keys are
    the request content only (messages, max tokens, temperature), as before the rebuild."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cassette: Cassette
    inner: BaseChatModel | None = None
    label: str

    @property
    def _llm_type(self) -> str:
        return "replay"

    async def aclose(self) -> None:
        self.cassette.save()
        await aclose_all(self.inner)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        key = chat_key(
            as_chat_messages(messages), kwargs.get("max_tokens", 0), kwargs.get("temperature", 0.0)
        )
        recorded = self.cassette.chat(key)
        if recorded is not None:
            return self._answer(recorded, self.label)
        if not self.cassette.record or self.inner is None:
            raise ReplayMissError(f"no recorded chat response {key[:12]}: {RERECORD}")
        result = await self.inner._agenerate(messages, stop, run_manager, **kwargs)
        if (entry := self._recording(result, self.label)) is not None:
            self.cassette.put_chat(key, entry)
        return result
