"""Evaluation runs as Langfuse experiments: dataset sync and the experiment run, offline."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse.api import DatasetItem, DatasetStatus
from opentelemetry.sdk.trace import TracerProvider

from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation
from customer_agent.evaluation.langfuse_experiment import (
    ACTIVE,
    ARCHIVED,
    CALIBRATION_QUEUE,
    HUMAN_SCORE,
    check_scores,
    ensure_calibration_queue,
    item_fields,
    item_id,
    labelled_rows,
    queue_for_labelling,
    run_experiment,
    sync_dataset,
)
from customer_agent.evaluation.schema import load_dataset
from customer_agent.evaluation.scoring import score_case
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


class FakeQueueApi:
    """The score-config and annotation-queue endpoints of the Langfuse API."""

    def __init__(self, configs: list[SimpleNamespace], queues: list[SimpleNamespace]) -> None:
        self.created: list[str] = []
        self.queued: list[tuple[str, str]] = []
        self.score_configs = SimpleNamespace(
            get=lambda **_: SimpleNamespace(data=configs), create=self._create_config
        )
        self.annotation_queues = SimpleNamespace(
            list_queues=lambda **_: SimpleNamespace(data=queues),
            create_queue=self._create_queue,
            create_queue_item=self._queue,
        )

    def _create_config(self, **fields: Any) -> SimpleNamespace:
        self.created.append(f"config:{fields['name']}")
        return SimpleNamespace(id="config-1", name=fields["name"], is_archived=False)

    def _create_queue(self, **fields: Any) -> SimpleNamespace:
        self.created.append(f"queue:{fields['name']}:{','.join(fields['score_config_ids'])}")
        return SimpleNamespace(id="queue-1", name=fields["name"])

    def _queue(self, queue_id: str, **fields: Any) -> None:
        self.queued.append((queue_id, fields["object_id"]))


def test_the_calibration_queue_is_created_once_and_then_reused() -> None:
    fresh = SimpleNamespace(api=FakeQueueApi([], []))
    existing = SimpleNamespace(
        api=FakeQueueApi(
            [SimpleNamespace(id="config-9", name=HUMAN_SCORE, is_archived=False)],
            [SimpleNamespace(id="queue-9", name=CALIBRATION_QUEUE)],
        )
    )

    assert ensure_calibration_queue(fresh) == "queue-1"  # type: ignore[arg-type]
    assert fresh.api.created == [
        f"config:{HUMAN_SCORE}",
        f"queue:{CALIBRATION_QUEUE}:config-1",
    ]
    assert ensure_calibration_queue(existing) == "queue-9"  # type: ignore[arg-type]
    assert existing.api.created == []


def test_only_judged_cases_with_a_trace_are_queued() -> None:
    response = AgentResponse(answer="Nein.", behavior=Behavior.ANSWER)
    judged, unjudged, untraced = (score_case(case, response) for case in CASES)
    results = [
        judged.model_copy(update={"grounded": True}),
        unjudged,
        untraced.model_copy(update={"grounded": False}),
    ]
    langfuse = SimpleNamespace(api=FakeQueueApi([], []))

    queued = queue_for_labelling(
        langfuse,  # type: ignore[arg-type]
        "queue-1",
        results,
        {judged.case_id: "trace-1", unjudged.case_id: "trace-2"},
    )

    assert queued == 1
    assert langfuse.api.queued == [("queue-1", "trace-1")]


def test_human_labels_become_calibration_rows_next_to_the_judges_verdict() -> None:
    label = SimpleNamespace(
        subject=SimpleNamespace(kind="trace", id="trace-1"),
        value=0.0,
        timestamp=datetime(2026, 10, 23, tzinfo=UTC),
    )
    root = SimpleNamespace(
        input={"turns": [{"role": "user", "content": "Kostet die Rücksendung etwas?"}]},
        output={"answer": "Nein, die Rücksendung ist kostenlos.", "sources": ["policy-returns-v5"]},
        metadata={"case_id": "N-002"},
    )

    def scores(**query: Any) -> SimpleNamespace:
        data = [label] if "queue_id" in query else [SimpleNamespace(value="True")]  # the judge
        return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=None))

    langfuse = SimpleNamespace(
        api=SimpleNamespace(
            scores=SimpleNamespace(get_many=scores),
            observations=SimpleNamespace(get_many=lambda **_: SimpleNamespace(data=[root])),
        )
    )

    assert labelled_rows(langfuse, "queue-1") == [  # type: ignore[arg-type]
        {
            "case_id": "N-002",
            "question": "Kostet die Rücksendung etwas?",
            "answer": "Nein, die Rücksendung ist kostenlos.",
            "sources": ["policy-returns-v5"],
            "judge_grounded": True,
            "human_grounded": False,
            "trace_id": "trace-1",
        }
    ]
