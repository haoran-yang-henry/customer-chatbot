"""Request context as pure ASGI middleware.

Unlike ``@app.middleware("http")`` (BaseHTTPMiddleware), this wraps the whole response including
a streamed body: the request id is set for every log line written while streaming, and the
access log reports the real duration of a streamed answer, not just the time to the headers.
"""

import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from customer_agent.observability import elapsed_ms, get_logger, request_id_var

log = get_logger("http")

# Client-supplied ids end up in logs and response headers: accept only a safe, short format.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id_from(header: str | None) -> str:
    if header and _REQUEST_ID.fullmatch(header):
        return header
    return uuid.uuid4().hex[:12]


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"x-request-id")
        request_id = request_id_from(raw.decode("latin-1") if raw else None)
        status = 500
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        token = request_id_var.set(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "%s %s status=%d took=%dms",
                scope["method"],
                scope["path"],
                status,
                elapsed_ms(start),
            )
            request_id_var.reset(token)
