"""FastAPI application factory."""

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from customer_agent import __version__
from customer_agent.api import chat as chat_api
from customer_agent.api import health
from customer_agent.application.chat import ChatService
from customer_agent.config import Settings, get_settings
from customer_agent.observability import configure_logging, elapsed_ms, get_logger, request_id_var

log = get_logger("http")


def create_app(settings: Settings | None = None, chat: ChatService | None = None) -> FastAPI:
    """``chat`` lets tests inject a fake; otherwise the lifespan builds the real one."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if chat is not None:
            app.state.chat = chat
            yield
            return
        from customer_agent.bootstrap import build_container

        container = build_container(settings)
        app.state.chat = container.chat
        log.info("ready model=%s prompts=%s", container.llm_model, container.prompts)
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title="NordElektro Customer Service Agent",
        version=__version__,
        docs_url="/docs" if settings.app_env != "prod" else None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        log.info(
            "%s %s status=%d took=%dms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms(start),
            request_id,
        )
        return response

    app.include_router(health.router)
    app.include_router(chat_api.router)
    return app
