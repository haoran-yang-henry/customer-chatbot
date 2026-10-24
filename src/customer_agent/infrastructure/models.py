"""Chat models (rebuild phase 2, ADR-0015): LangChain's ``ChatOpenAI`` for every model role, with
the project's error contract and circuit breaker; evaluation runs and the CI gate put the
recorded-response models of ``recorded.py`` in front.

    ReplayChatModel (CI)  →  CachedChatModel (evaluation)  →  BreakerChatModel  →  ProviderChatModel

Each wrapper calls the next one's ``_agenerate`` / ``_astream`` directly, so LangChain's callbacks
see one model run per call, not one per layer. Callers pass ``max_tokens``, ``temperature`` and
the conversation's ``session_id`` as call keyword arguments; only ``ProviderChatModel`` knows that
the provider wants the session as a header.
"""

import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, cast

import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import LangSmithParams
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    UsageMetadata,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from pydantic import ConfigDict, model_validator

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.resilience import CircuitBreaker
from customer_agent.observability import elapsed_ms, get_logger, record_tokens, step, tracer
from customer_agent.observability.metrics import LLM_FIRST_TOKEN, STEP_DURATION

log = get_logger("llm")

# LangChain types streams as AsyncIterator; they are async generators, which can be closed.
Stream = AsyncGenerator[ChatGenerationChunk, None]


def model_label(model: BaseChatModel) -> str:
    """The provider's model name through any wrappers."""
    while isinstance(inner := getattr(model, "inner", None), BaseChatModel):
        model = inner
    return str(getattr(model, "label", None) or getattr(model, "model_name", model._llm_type))


class AsyncOnlyChatModel(BaseChatModel):
    """Base of the wrappers: async only (a synchronous call is a bug), and traced as the model
    they wrap, so Langfuse names the provider's model, also for streamed calls."""

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        inner = getattr(self, "inner", None)
        if isinstance(inner, BaseChatModel):
            return inner._get_ls_params(stop=stop, **kwargs)
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_model_name"] = model_label(self)  # replay without a provider: the recording's
        return params

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

    session_header: str | None = None
    """Per-conversation header (``x-opencode-session``, ADR-0008); OpenCode Go answers 400
    without it."""

    @model_validator(mode="before")
    @classmethod
    def _own_http_clients(cls, values: Any) -> Any:
        """Without them, every ``ChatOpenAI`` shares LangChain's process-wide HTTP client, and
        ``aclose`` would close it for every other model too (the evaluation's judge runs after
        the agent has shut down)."""
        if isinstance(values, dict):
            values.setdefault("http_client", openai.DefaultHttpxClient())
            values.setdefault("http_async_client", openai.DefaultAsyncHttpxClient())
        return values

    def _with_session(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Callers pass the conversation's id as the provider-neutral ``session_id`` keyword
        (LangChain's ``astream`` gives ``_astream`` no run manager, so run metadata cannot carry
        it); the endpoint wants it as a header, never in the request body."""
        kwargs = dict(kwargs)
        session = kwargs.pop("session_id", None)
        if self.session_header and session:
            headers = kwargs.get("extra_headers", {})
            kwargs["extra_headers"] = {**headers, self.session_header: str(session)}
        return kwargs

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
                result = await super()._agenerate(
                    messages, stop, run_manager, **self._with_session(kwargs)
                )
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
            kwargs = self._with_session(kwargs)
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


class BreakerChatModel(AsyncOnlyChatModel):
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
