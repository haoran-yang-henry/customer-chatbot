"""Chat models (rebuild phase 2, ADR-0015): LangChain's ``ChatOpenAI`` for every model role, with
the project's resilience, evaluation cache and CI replay as ``BaseChatModel`` wrappers.

    ReplayChatModel (CI)  →  CachedChatModel (evaluation)  →  BreakerChatModel  →  ProviderChatModel

Each wrapper calls the next one's ``_agenerate`` / ``_astream`` directly, so LangChain's callbacks
see one model run per call, not one per layer. Request parameters (``max_tokens``,
``temperature``, the session header in ``extra_headers``) travel as call keyword arguments, the
way ``model.ainvoke(messages, max_tokens=…)`` passes them.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Any, cast

import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    UsageMetadata,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from pydantic import ConfigDict

from customer_agent.application.ports import ChatMessage, LLMResult, ProviderUnavailableError
from customer_agent.infrastructure.cassette import RERECORD, Cassette, ReplayMissError, chat_key
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.resilience import CircuitBreaker
from customer_agent.observability import elapsed_ms, get_logger, record_tokens, step, tracer
from customer_agent.observability.metrics import LLM_FIRST_TOKEN, STEP_DURATION

log = get_logger("llm")

# LangChain types streams as AsyncIterator; they are async generators, which can be closed.
Stream = AsyncGenerator[ChatGenerationChunk, None]

_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}


def as_chat_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """The project's message shape: keys of the cassette and the cache are built from it."""
    return [ChatMessage(role=_ROLES[type(m)], content=m.text) for m in messages]  # type: ignore[arg-type]


def as_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    kinds = {role: kind for kind, role in _ROLES.items()}
    return [kinds[m.role](content=m.content) for m in messages]


def model_label(model: BaseChatModel) -> str:
    """The provider's model name through any wrappers."""
    while isinstance(inner := getattr(model, "inner", None), BaseChatModel):
        model = inner
    return str(getattr(model, "label", None) or getattr(model, "model_name", model._llm_type))


def _usage(prompt: int | None, completion: int | None) -> UsageMetadata | None:
    if prompt is None or completion is None:
        return None
    return UsageMetadata(
        input_tokens=prompt, output_tokens=completion, total_tokens=prompt + completion
    )


def _result(text: str, usage: UsageMetadata | None) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=usage))]
    )


class _AsyncOnly(BaseChatModel):
    """The agent only calls models asynchronously; a synchronous call is a bug."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(f"{type(self).__name__} is async only")


class ProviderChatModel(ChatOpenAI):
    """``ChatOpenAI`` against any OpenAI-compatible endpoint (ADR-0008), with the project's error
    contract (``ProviderUnavailableError``), token and latency metrics, and two corrections:

    - reasoning models get no ``temperature`` (the endpoint rejects or ignores it);
    - streamed usage is counted once: some endpoints (OpenCode Go) report it on the finishing
      chunk and again on a usage-only chunk, and LangChain adds chunks up.
    """

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.reasoning_effort:
            payload.pop("temperature", None)
        return payload

    async def aclose(self) -> None:
        try:
            if self.root_async_client is not None:
                await self.root_async_client.close()
        finally:
            if self.root_client is not None:
                self.root_client.close()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        start = time.perf_counter()
        with step("llm", model=self.model_name) as span:
            try:
                result = await super()._agenerate(messages, stop, run_manager, **kwargs)
            except openai.OpenAIError as exc:
                log.warning(
                    "model=%s error=%s took=%dms",
                    self.model_name,
                    type(exc).__name__,
                    elapsed_ms(start),
                )
                raise ProviderUnavailableError(f"{self.model_name}: {exc}") from exc
            message = result.generations[0].message
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            prompt = usage["input_tokens"] if usage else None
            completion = usage["output_tokens"] if usage else None
            span.set_attributes({"tokens_in": prompt or 0, "tokens_out": completion or 0})
        if not message.text:
            # Reasoning models can spend the whole token budget before writing any output.
            log.warning(
                "model=%s empty output finish_reason=%s tokens_out=%s",
                self.model_name,
                message.response_metadata.get("finish_reason"),
                completion,
            )
        record_tokens(self.model_name, prompt, completion)
        log.info(
            "model=%s tokens_in=%s tokens_out=%s took=%dms",
            self.model_name,
            prompt,
            completion,
            elapsed_ms(start),
        )
        return result

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        start = time.perf_counter()
        chars, finish_reason, outcome = 0, None, "ok"
        usage: UsageMetadata | None = None
        # Not attached as the current span: this generator yields to its caller, and a context
        # attached here would be detached elsewhere (see observability.step).
        span = tracer.start_span("agent.llm.stream", attributes={"model": self.model_name})
        try:
            stream = cast(Stream, super()._astream(messages, stop, run_manager, **kwargs))
            async with aclosing(stream) as chunks:  # an early close releases the connection
                async for chunk in chunks:
                    message = chunk.message
                    if isinstance(message, AIMessageChunk) and message.usage_metadata:
                        usage = message.usage_metadata  # the last report is the complete one
                        message = message.model_copy(update={"usage_metadata": None})
                        chunk = ChatGenerationChunk(
                            message=message, generation_info=chunk.generation_info
                        )
                    info = chunk.generation_info or {}
                    finish_reason = info.get("finish_reason") or finish_reason
                    if content := chunk.text:
                        if chars == 0:
                            first = time.perf_counter() - start
                            LLM_FIRST_TOKEN.labels(model=self.model_name).observe(first)
                            span.set_attribute("time_to_first_token_ms", round(first * 1000))
                        chars += len(content)
                    yield chunk
        except openai.OpenAIError as exc:
            log.warning("model=%s stream error=%s", self.model_name, type(exc).__name__)
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR)
            outcome = "error"
            raise ProviderUnavailableError(f"{self.model_name}: {exc}") from exc
        finally:
            span.end()
            STEP_DURATION.labels(step="llm_stream", outcome=outcome).observe(
                time.perf_counter() - start
            )
        if usage is not None:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))
        if chars == 0:
            log.warning("model=%s empty stream finish_reason=%s", self.model_name, finish_reason)
        prompt = usage["input_tokens"] if usage else None
        completion = usage["output_tokens"] if usage else None
        record_tokens(self.model_name, prompt, completion)
        log.info(
            "model=%s stream tokens_in=%s tokens_out=%s took=%dms",
            self.model_name,
            prompt,
            completion,
            elapsed_ms(start),
        )


class BreakerChatModel(_AsyncOnly):
    """Every call passes the circuit breaker of its endpoint (Day 7, NFR-11)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseChatModel
    breaker: CircuitBreaker

    @property
    def _llm_type(self) -> str:
        return "breaker"

    async def aclose(self) -> None:
        await aclose_all(self.inner)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.breaker.before_call()
        try:
            result = await self.inner._agenerate(messages, stop, run_manager, **kwargs)
        except ProviderUnavailableError:
            self.breaker.record_failure()
            raise
        except BaseException:
            self.breaker.abandon()
            raise
        self.breaker.record_success()
        return result

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.breaker.before_call()
        finished = False
        try:
            inner = cast(Stream, self.inner._astream(messages, stop, run_manager, **kwargs))
            async with aclosing(inner) as chunks:
                async for chunk in chunks:
                    yield chunk
            finished = True
        except ProviderUnavailableError:
            self.breaker.record_failure()
            raise
        finally:
            if finished:
                self.breaker.record_success()
            else:
                self.breaker.abandon()  # no-op after record_failure


class CachedChatModel(_AsyncOnly):
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
            cached = LLMResult.model_validate_json(
                await asyncio.to_thread(path.read_text, encoding="utf-8")
            )
            # Count the recorded usage: an evaluation's token totals are what the run would cost.
            record_tokens(model_label(self.inner), cached.prompt_tokens, cached.completion_tokens)
            return _result(cached.text, _usage(cached.prompt_tokens, cached.completion_tokens))
        result = await self.inner._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        if message.text.strip():  # never cache an empty (failed) generation
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            entry = LLMResult(
                text=message.text,
                model=model_label(self.inner),
                prompt_tokens=usage["input_tokens"] if usage else None,
                completion_tokens=usage["output_tokens"] if usage else None,
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, entry.model_dump_json(), encoding="utf-8")
        return result

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


class ReplayChatModel(_AsyncOnly):
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
            record_tokens(self.label, recorded.prompt_tokens, recorded.completion_tokens)
            return _result(
                recorded.text, _usage(recorded.prompt_tokens, recorded.completion_tokens)
            )
        if not self.cassette.record or self.inner is None:
            raise ReplayMissError(f"no recorded chat response {key[:12]}: {RERECORD}")
        result = await self.inner._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        if message.text.strip():  # an empty generation is a failure, not a recording
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            self.cassette.put_chat(
                key,
                LLMResult(
                    text=message.text,
                    model=self.label,
                    prompt_tokens=usage["input_tokens"] if usage else None,
                    completion_tokens=usage["output_tokens"] if usage else None,
                ),
            )
        return result

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


class ChatModelLLM:
    """``LLMProvider`` over a chat model: the bridge while the application still calls the
    port. Removed in rebuild phase 2 once router, classifier, answer and judge use chat models.
    """

    def __init__(self, model: BaseChatModel, *, session_header: str | None = None) -> None:
        self._model = model
        self._session_header = session_header

    @property
    def model_name(self) -> str:
        return model_label(self._model)

    @property
    def chat_model(self) -> BaseChatModel:
        return self._model

    async def aclose(self) -> None:
        await aclose_all(self._model)

    def _kwargs(
        self, max_tokens: int, temperature: float, session_id: str | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if self._session_header and session_id:
            kwargs["extra_headers"] = {self._session_header: session_id}
        return kwargs

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        reply = await self._model.ainvoke(
            as_langchain(messages), **self._kwargs(max_tokens, temperature, session_id)
        )
        usage = reply.usage_metadata if isinstance(reply, AIMessage) else None
        return LLMResult(
            text=reply.text,
            model=self.model_name,
            prompt_tokens=usage["input_tokens"] if usage else None,
            completion_tokens=usage["output_tokens"] if usage else None,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        stream = self._model.astream(
            as_langchain(messages), **self._kwargs(max_tokens, temperature, session_id)
        )
        async with aclosing(cast(AsyncGenerator[BaseMessageChunk, None], stream)) as chunks:
            async for chunk in chunks:
                if chunk.text:
                    yield chunk.text
