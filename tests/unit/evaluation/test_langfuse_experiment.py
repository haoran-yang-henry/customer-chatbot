"""Evaluation runs as Langfuse experiments: dataset sync and the experiment run, offline."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse.api import DatasetItem, DatasetStatus
from opentelemetry.sdk.trace import TracerProvider

from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation
from customer_agent.evaluation.langfuse_experiment import (
    ACTIVE,
    ARCHIVED,
    check_scores,
    item_fields,
    item_id,
    run_experiment,
    sync_dataset,
)
from customer_agent.evaluation.schema import load_dataset
from tests.conftest import CORE_DATASET

CASES = load_dataset(CORE_DATASET)[:3]


def dataset_item(
    case_id: str, fields: dict[str, Any], status: DatasetStatus = ACTIVE
) -> DatasetItem:
    now = datetime.now(UTC)
    return DatasetItem(
        id=item_id("core", case_id),
        status=status,
        dataset_id="dataset-1",
        dataset_name="core",
        source_trace_id=None,
        source_observation_id=None,
        created_at=now,
        updated_at=now,
        version=None,
        media_references=[],
        **fields,
    )


class FakeDatasets:
    """The dataset calls of the Langfuse client that ``sync_dataset`` makes."""

    def __init__(self, items: list[DatasetItem]) -> None:
        self.items = items
        self.written: list[tuple[str, DatasetStatus]] = []

    def create_dataset(self, **_: Any) -> None:
        pass

    def get_dataset(self, name: str) -> "FakeDatasets":
        return self

    def create_dataset_item(self, **fields: Any) -> None:
        self.written.append((fields["id"], fields["status"]))


def test_sync_writes_changed_and_new_cases_and_archives_removed_ones() -> None:
    unchanged, changed, new = CASES
    langfuse = FakeDatasets(
        [
            dataset_item(unchanged.id, item_fields(unchanged)),
            dataset_item(changed.id, item_fields(changed) | {"metadata": {"case_id": changed.id}}),
            dataset_item("N-999", item_fields(unchanged)),  # no longer in the file
        ]
    )

    written = sync_dataset(langfuse, "core", CORE_DATASET, CASES)  # type: ignore[arg-type]

    assert written == 3
    assert sorted(langfuse.written) == sorted(
        [
            (item_id("core", changed.id), ACTIVE),
            (item_id("core", new.id), ACTIVE),
            (item_id("core", "N-999"), ARCHIVED),
        ]
    )


def test_every_check_becomes_a_score() -> None:
    output = {"checks": {"behavior": True, "intent": False}, "passed": False, "source_recall": 0.5}

    scores = check_scores(output=output)

    assert {(score.name, score.value) for score in scores} == {
        ("check_behavior", True),
        ("check_intent", False),
        ("passed", False),
        ("source_recall", 0.5),
    }


class ThreadRecordingAgent:
    """Answers like the placeholder agent and records the thread it runs on."""

    def __init__(self) -> None:
        self.threads: set[int] = set()

    async def respond(self, conversation: Conversation) -> AgentResponse:
        self.threads.add(threading.get_ident())
        return AgentResponse(
            answer="Dazu kann ich Ihnen leider noch keine Auskunft geben.",
            behavior=Behavior.NO_ANSWER,
        )


@pytest.fixture
def offline_langfuse() -> Iterator[Langfuse]:
    client = Langfuse(
        public_key="pk-lf-eval-test",
        secret_key="sk-lf-eval-test",
        base_url="http://127.0.0.1:9",  # nothing listens here: traces are dropped
        tracer_provider=TracerProvider(),
        timeout=1,
    )
    yield client
    client.shutdown()


async def test_experiment_runs_the_agent_on_the_evaluation_loop_and_keeps_case_order(
    offline_langfuse: Langfuse,
) -> None:
    agent = ThreadRecordingAgent()
    cases = list(reversed(CASES))

    results, traces, _ = await run_experiment(
        offline_langfuse,
        dataset="core",
        items=[dataset_item(case.id, item_fields(case)) for case in CASES],
        cases=cases,
        agent=agent,
        current_day=8,
        run_name="offline",
        metadata={},
        concurrency=2,
    )

    assert [result.case_id for result in results] == [case.id for case in cases]
    assert set(traces) == {case.id for case in CASES}
    # The agent's pool and clients belong to the evaluation's loop, not Langfuse's thread.
    assert agent.threads == {threading.get_ident()}
