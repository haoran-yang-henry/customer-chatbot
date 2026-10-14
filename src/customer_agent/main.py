"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from customer_agent import __version__
from customer_agent.api import chat as chat_api
from customer_agent.api import health, metrics
from customer_agent.api.middleware import RequestContextMiddleware
from customer_agent.api.problems import register_problem_handlers
from customer_agent.api.rate_limit import SlidingWindowLimiter
from customer_agent.application.chat import ChatService
from customer_agent.application.ports import TokenVerifier
from customer_agent.config import Settings, get_settings
from customer_agent.observability import configure_logging, configure_tracing, get_logger

log = get_logger("http")


def create_app(
    settings: Settings | None = None,
    chat: ChatService | None = None,
    verifier: TokenVerifier | None = None,
) -> FastAPI:
    """``chat``/``verifier`` let tests inject fakes; otherwise the lifespan builds real ones."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    tracing = configure_tracing("customer-agent", __version__, settings.otel_endpoint)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tenant_id = settings.tenant_id
        app.state.limiter = (
            SlidingWindowLimiter(settings.rate_limit_per_minute)
            if settings.rate_limit_per_minute > 0
            else None
        )
        app.state.readiness = None
        if chat is not None:
            app.state.chat = chat
            app.state.verifier = verifier
            yield
            return
        from customer_agent.bootstrap import build_container

        container = build_container(settings)
        app.state.chat = container.chat
        app.state.verifier = container.verifier
        app.state.readiness = container.readiness
        if container.verifier is None:
            log.warning("authentication disabled (CA_AUTH_ISSUER unset): all requests anonymous")
        log.info("ready models=%s prompts=%s", container.models, container.prompts)
        try:
            yield
        finally:
            await container.aclose()
            tracing.force_flush()  # spans still queued in the batch processor

    app = FastAPI(
        title="NordElektro Customer Service Agent",
        version=__version__,
        docs_url="/docs" if settings.app_env != "prod" else None,
        lifespan=lifespan,
        # FastAPI's native tracing provides the server span (route-named, W3C propagation);
        # exporters, metrics and logs stay configured in one place: customer_agent.observability.
        telemetry={"auto_configure": False, "metrics": False, "logs": False},
    )
    app.add_middleware(RequestContextMiddleware)
    register_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(chat_api.router)
    return app
