import json

from fastapi.testclient import TestClient

from customer_agent.application.answering import FinalAnswer, StreamEvent, TextDelta
from customer_agent.config import Settings
from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation
from customer_agent.main import create_app

ANSWER = AgentResponse(
    answer="Die Rücksendung ist kostenlos.", behavior=Behavior.ANSWER, sources=["policy-returns-v5"]
)


class FakeAgent:
    def __init__(self) -> None:
        self.received: list[Conversation] = []

    async def respond(self, conversation: Conversation) -> AgentResponse:
        self.received.append(conversation)
        return ANSWER

    async def respond_stream(self, conversation: Conversation):  # type: ignore[no-untyped-def]
        self.received.append(conversation)
        events: list[StreamEvent] = [
            TextDelta(text="Die Rücksendung "),
            TextDelta(text="ist kostenlos."),
            FinalAnswer(response=ANSWER),
        ]
        for event in events:
            yield event


def client(agent: FakeAgent) -> TestClient:
    return TestClient(create_app(Settings(app_env="test"), agent=agent))  # type: ignore[arg-type]


def test_chat_returns_answer_sources_and_request_id() -> None:
    agent = FakeAgent()
    with client(agent) as c:
        response = c.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Kostet die Rücksendung etwas?"}]},
            headers={"x-request-id": "req-123"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Die Rücksendung ist kostenlos.",
        "behavior": "answer",
        "sources": ["policy-returns-v5"],
        "request_id": "req-123",
    }
    assert response.headers["x-request-id"] == "req-123"
    assert agent.received[0].customer_id is None


def test_last_message_must_be_from_the_user() -> None:
    with client(FakeAgent()) as c:
        response = c.post("/v1/chat", json={"messages": [{"role": "assistant", "content": "Hi"}]})

    assert response.status_code == 422


def test_empty_message_list_is_rejected() -> None:
    with client(FakeAgent()) as c:
        assert c.post("/v1/chat", json={"messages": []}).status_code == 422


def test_stream_sends_deltas_then_final_event() -> None:
    with (
        client(FakeAgent()) as c,
        c.stream(
            "POST", "/v1/chat/stream", json={"messages": [{"role": "user", "content": "Kosten?"}]}
        ) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode()

    events = [block.split("\n") for block in body.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    payloads = [json.loads(lines[1].removeprefix("data: ")) for lines in events]

    assert names == ["delta", "delta", "final"]
    assert "".join(p["text"] for p in payloads[:2]) == "Die Rücksendung ist kostenlos."
    assert payloads[2]["sources"] == ["policy-returns-v5"]
