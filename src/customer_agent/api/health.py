"""Probes (Day 7): liveness says the process runs, readiness says it can serve traffic."""

from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from customer_agent import __version__

router = APIRouter(tags=["operations"])

Readiness = Callable[[], Awaitable[dict[str, str]]]
REQUIRED = ("database", "checkpoints")  # checks that decide readiness


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    version: str
    checks: dict[str, str]


@router.get("/health")
@router.get("/health/live")
async def health() -> HealthResponse:
    """Liveness: no dependencies are checked, so a database outage never restarts the process."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/health/ready", responses={503: {"model": ReadinessResponse}})
async def ready(request: Request, response: Response) -> ReadinessResponse:
    """Readiness: the database must answer and the checkpoint tables exist. Model providers are
    reported but not required, because the agent degrades to the outage fallback without them."""
    readiness: Readiness | None = getattr(request.app.state, "readiness", None)
    checks = await readiness() if readiness is not None else {}
    if any(checks.get(name, "ok") != "ok" for name in REQUIRED):
        response.status_code = 503
        return ReadinessResponse(status="unavailable", version=__version__, checks=checks)
    return ReadinessResponse(status="ready", version=__version__, checks=checks)
