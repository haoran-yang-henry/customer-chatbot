"""OpenAI-compatible chat completions adapter (OpenAI, OpenCode Go, Azure OpenAI, …), ADR-0008."""

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
from opentelemetry import trace

from customer_agent.application.ports import ChatMessage, LLMResult, ProviderUnavailableError
from customer_agent.observability import elapsed_ms, get_logger, record_tokens, step, tracer
from customer_agent.observability.metrics import LLM_FIRST_TOKEN

log = get_logger("llm")


class OpenAICompatibleLLM:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        session_header: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._session_header = session_header
        self._reasoning_effort = reasoning_effort

    @property
    def model_name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.close()

    def _request(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        temperature: float,
        session_id: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._reasoning_effort:
            # Reasoning models reject max_tokens/temperature and need the newer parameters.
            request.pop("temperature")
            request["max_completion_tokens"] = request.pop("max_tokens")
            request["reasoning_effort"] = self._reasoning_effort
        if self._session_header and session_id:
            request["extra_headers"] = {self._session_header: session_id}
        return request

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        with step("llm", model=self._model) as span:
            result = await self._complete(messages, max_tokens, temperature, session_id)
            span.set_attributes(
                {
                    "tokens_in": result.prompt_tokens or 0,
                    "tokens_out": result.completion_tokens or 0,
                }
            )
            return result

    async def _complete(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        temperature: float,
        session_id: str | None,
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, session_id)
            )
        except openai.OpenAIError as exc:
            log.warning(
                "model=%s error=%s took=%dms", self._model, type(exc).__name__, elapsed_ms(start)
            )
            raise ProviderUnavailableError(f"{self._model}: {exc}") from exc
        usage = response.usage
        if not response.choices[0].message.content:
            # Reasoning models can spend the whole token budget before writing any output.
            log.warning(
                "model=%s empty output finish_reason=%s tokens_out=%s",
                self._model,
                response.choices[0].finish_reason,
                usage.completion_tokens if usage else None,
            )
        result = LLMResult(
            text=response.choices[0].message.content or "",
            model=response.model or self._model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        record_tokens(self._model, result.prompt_tokens, result.completion_tokens)
        log.info(
            "model=%s tokens_in=%s tokens_out=%s took=%dms",
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            elapsed_ms(start),
        )
        return result

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        chars, finish_reason, usage = 0, None, None
        # Not attached as the current span: this generator yields to its caller, and a context
        # attached here would be detached elsewhere (see observability.step).
        span = tracer.start_span("agent.llm.stream", attributes={"model": self._model})
        try:
            chunks = await self._client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, session_id),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in chunks:
                usage = chunk.usage or usage  # sent in a final chunk without choices
                if not chunk.choices:
                    continue
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                if content := chunk.choices[0].delta.content:
                    if chars == 0:
                        first = time.perf_counter() - start
                        LLM_FIRST_TOKEN.labels(model=self._model).observe(first)
                        span.set_attribute("time_to_first_token_ms", round(first * 1000))
                    chars += len(content)
                    yield content
        except openai.OpenAIError as exc:
            log.warning("model=%s stream error=%s", self._model, type(exc).__name__)
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR)
            raise ProviderUnavailableError(f"{self._model}: {exc}") from exc
        finally:
            span.end()
        if chars == 0:
            log.warning("model=%s empty stream finish_reason=%s", self._model, finish_reason)
        record_tokens(
            self._model,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )
        log.info(
            "model=%s stream tokens_in=%s tokens_out=%s took=%dms",
            self._model,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            elapsed_ms(start),
        )
