import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from customer_agent.application.answering import TextDelta
from customer_agent.application.chat import ChatTurn, SessionNotFoundError
from customer_agent.application.ports import AuthenticationError, ConcurrentTurnError, Principal
from customer_agent.config import Settings
from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent
from customer_agent.main import create_app

TURN = ChatTurn(
    session_id="abc",
    response=AgentResponse(
        answer="Die Rücksendung ist kostenlos.",
        behavior=Behavior.ANSWER,
        intent=Intent.RETURN_POLICY,
        sources=["policy-returns-v5"],
    ),
)


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None, str]] = []

    async def chat_stream(
        self, session_id: str | None, customer_id: str | None, text: str
    ) -> AsyncIterator[TextDelta | ChatTurn]:
        self.calls.append((session_id, customer_id, text))
        if session_id == "missing":
            raise SessionNotFoundError(session_id)
        if session_id == "busy":
            raise ConcurrentTurnError(session_id)
        yield TextDelta(text="Die Rücksendung ")
        yield TextDelta(text="ist kostenlos.")
        yield TURN

    async def chat(self, session_id: str | None, customer_id: str | None, text: str) -> ChatTurn:
        async for event in self.chat_stream(session_id, customer_id, text):
            if isinstance(event, ChatTurn):
                return event
        raise AssertionError


class FakeVerifier:
    async def verify(self, token: str) -> Principal:
        if token == "anna":
            return Principal(subject="u1", customer_id="C-1001", tenant_id="nordelektro")
        if token == "other-tenant":
            return Principal(subject="u2", customer_id="X-1", tenant_id="other-shop")
        raise AuthenticationError("bad token")


def client(chat: FakeChat, verifier: FakeVerifier | None = None) -> TestClient:
    app = create_app(Settings(app_env="test"), chat=chat, verifier=verifier)  # type: ignore[arg-type]
    return TestClient(app)


def test_chat_returns_session_answer_intent_and_request_id() -> None:
    chat = FakeChat()
    with client(chat) as c:
        response = c.post(
            "/v1/chat",
            json={"message": "Kostet die Rücksendung etwas?"},
            headers={"x-request-id": "req-123"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "abc",
        "answer": "Die Rücksendung ist kostenlos.",
        "behavior": "answer",
        "intent": "RETURN_POLICY",
        "sources": ["policy-returns-v5"],
        "request_id": "req-123",
    }
    assert chat.calls == [(None, None, "Kostet die Rücksendung etwas?")]  # customer from token only


def test_unknown_session_is_404_for_both_endpoints() -> None:
    with client(FakeChat()) as c:
        body = {"session_id": "missing", "message": "Hallo"}
        assert c.post("/v1/chat", json=body).status_code == 404
        assert c.post("/v1/chat/stream", json=body).status_code == 404


def test_empty_message_is_rejected() -> None:
    with client(FakeChat()) as c:
        assert c.post("/v1/chat", json={"message": ""}).status_code == 422


def test_stream_sends_deltas_then_final_event() -> None:
    with (
        client(FakeChat()) as c,
        c.stream("POST", "/v1/chat/stream", json={"message": "Kosten?"}) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode()

    events = [block.split("\n") for block in body.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    payloads = [json.loads(lines[1].removeprefix("data: ")) for lines in events]

    assert names == ["delta", "delta", "final"]
    assert payloads[2]["session_id"] == "abc" and payloads[2]["intent"] == "RETURN_POLICY"


def test_concurrent_turn_is_409() -> None:
    with client(FakeChat()) as c:
        response = c.post("/v1/chat", json={"session_id": "busy", "message": "Hallo"})

    assert response.status_code == 409


def test_errors_are_rfc9457_problem_details() -> None:
    with client(FakeChat()) as c:
        response = c.post(
            "/v1/chat",
            json={"session_id": "missing", "message": "Hallo"},
            headers={"x-request-id": "req-9"},
        )

    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "session not found",
        "request_id": "req-9",
    }


def test_validation_errors_list_the_offending_fields() -> None:
    with client(FakeChat()) as c:
        body = c.post("/v1/chat", json={"message": ""}).json()

    assert body["status"] == 422 and body["errors"][0]["loc"] == ["body", "message"]


def test_verified_token_identifies_the_customer() -> None:
    chat = FakeChat()
    with client(chat, FakeVerifier()) as c:
        response = c.post(
            "/v1/chat", json={"message": "Hallo"}, headers={"authorization": "Bearer anna"}
        )

    assert response.status_code == 200
    assert chat.calls == [(None, "C-1001", "Hallo")]


@pytest.mark.parametrize(
    ("header", "status"),
    [
        ("Bearer forged", 401),
        ("Basic dXNlcjpwYXNz", 401),
        ("Bearer", 401),
        ("Bearer other-tenant", 403),
    ],
)
def test_bad_or_foreign_tokens_are_rejected(header: str, status: int) -> None:
    chat = FakeChat()
    with client(chat, FakeVerifier()) as c:
        response = c.post("/v1/chat", json={"message": "Hallo"}, headers={"authorization": header})

    assert response.status_code == status
    assert chat.calls == []  # the agent never runs for a rejected request
    if status == 401:
        assert response.headers["www-authenticate"].startswith("Bearer")


def test_token_without_configured_authentication_is_rejected() -> None:
    with client(FakeChat(), verifier=None) as c:
        response = c.post(
            "/v1/chat", json={"message": "Hallo"}, headers={"authorization": "Bearer anna"}
        )

    assert response.status_code == 401
