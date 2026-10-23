"""Evaluation runs as Langfuse experiments (rebuild plan phase 1, ``--langfuse``).

The datasets in git stay the source of truth. Before a run, each file is synced to the Langfuse
dataset of the same name: changed cases are written, cases no longer in the file are archived.
The selected cases then run through Langfuse's experiment runner, so every case becomes a trace
linked to its dataset item, with one score per deterministic check; the judge's verdicts are
added to the same traces afterwards. The report is built from the same results as without
Langfuse.

Langfuse runs an experiment on an event loop of its own in a worker thread, while the agent's
database pool and model clients belong to the evaluation's loop. Each case is therefore sent
back to that loop, together with the experiment's trace context, so the agent's steps and model
calls appear under the case's trace.

Judge calibration: judged answers go to an annotation queue, a person labels them in Langfuse,
and the labels come back as a calibration sheet for ``--check-calibration``.
"""

import asyncio
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langfuse.api import (
    AnnotationQueueObjectType,
    DatasetItem,
    DatasetStatus,
    ScoreConfigDataType,
)
from langfuse.experiment import Evaluation, ExperimentItem, ExperimentItemResult
from opentelemetry import context as otel_context

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import AgentResponse
from customer_agent.evaluation.schema import EvalCase
from customer_agent.evaluation.scoring import CaseResult, score_case

# Langfuse's generated enums type their members as plain str; looked up by value they are typed.
ACTIVE = DatasetStatus("ACTIVE")
ARCHIVED = DatasetStatus("ARCHIVED")

JUDGE_SCORE = "grounded"
HUMAN_SCORE = "grounded_human"
CALIBRATION_QUEUE = "groundedness-calibration"


def item_id(dataset: str, case_id: str) -> str:
    """Dataset item ids are unique per Langfuse project, case ids only per file."""
    return f"{dataset}-{case_id}"


def item_fields(case: EvalCase) -> dict[str, Any]:
    data = case.model_dump(mode="json")
    expected = ("expected_behavior", "expected_intent", "expected_sources", "expected_tool")
    return {
        "input": {"turns": data["turns"], "customer_id": data["customer_id"]},
        "expected_output": {key.removeprefix("expected_"): data[key] for key in expected}
        | {"must_contain": data["must_contain"], "must_not_contain": data["must_not_contain"]},
        "metadata": {
            "case_id": case.id,
            "category": data["category"],
            "target_day": case.target_day,
            "tags": case.tags,
            "notes": case.notes,
        },
    }


def sync_dataset(client: Langfuse, name: str, source: Path, cases: Sequence[EvalCase]) -> int:
    """Make the Langfuse dataset match the file; returns the number of items written."""
    client.create_dataset(
        name=name,
        description=f"Synced from {source} before every run; edit the file, not this dataset.",
        metadata={"source": str(source)},
    )
    existing = {item.id: item for item in client.get_dataset(name).items}
    wanted = {item_id(name, case.id): item_fields(case) for case in cases}
    written = 0
    for id_, fields in wanted.items():
        item = existing.get(id_)
        current = None
        if item is not None and item.status == ACTIVE:
            current = {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
            }
        if current != fields:
            client.create_dataset_item(dataset_name=name, id=id_, status=ACTIVE, **fields)
            written += 1
    for id_, item in existing.items():
        if id_ not in wanted and item.status == ACTIVE:
            client.create_dataset_item(
                dataset_name=name,
                id=id_,
                status=ARCHIVED,
                input=item.input,
                expected_output=item.expected_output,
                metadata=item.metadata,
            )
            written += 1
    return written


def check_scores(*, output: dict[str, Any] | None, **_: Any) -> list[Evaluation]:
    """One boolean score per deterministic check of ``score_case``, plus the verdict."""
    if not output:
        return []
    scores = [
        Evaluation(name=f"check_{name}", value=ok, data_type="BOOLEAN")
        for name, ok in output["checks"].items()
    ]
    scores.append(Evaluation(name="passed", value=output["passed"], data_type="BOOLEAN"))
    if output["source_recall"] is not None:
        scores.append(
            Evaluation(name="source_recall", value=output["source_recall"], data_type="NUMERIC")
        )
    return scores


def pass_rate(*, item_results: list[ExperimentItemResult], **_: Any) -> Evaluation:
    passed = [bool(r.output["passed"]) for r in item_results if r.output]
    rate = sum(passed) / len(passed) if passed else 0.0
    return Evaluation(name="pass_rate", value=rate, data_type="NUMERIC")


async def run_experiment(
    client: Langfuse,
    *,
    dataset: str,
    items: Sequence[DatasetItem],
    cases: Sequence[EvalCase],
    agent: CustomerServiceAgent,
    current_day: int,
    run_name: str,
    metadata: dict[str, str],
    concurrency: int,
) -> tuple[list[CaseResult], dict[str, str], str | None]:
    """Results in case order, the trace id per case id, and the experiment's URL."""
    loop = asyncio.get_running_loop()
    by_item = {item_id(dataset, case.id): case for case in cases}
    results: dict[str, CaseResult] = {}

    async def respond(case: EvalCase, context: otel_context.Context) -> AgentResponse:
        token = otel_context.attach(context)  # the case's trace, on the agent's loop
        try:
            return await agent.respond(case.to_conversation())
        finally:
            otel_context.detach(token)

    async def task(*, item: ExperimentItem, **_: Any) -> dict[str, Any]:
        if not isinstance(item, DatasetItem):  # only dataset items are passed in below
            raise TypeError(f"expected a dataset item, got {type(item).__name__}")
        case = by_item[item.id]
        future = asyncio.run_coroutine_threadsafe(respond(case, otel_context.get_current()), loop)
        result = score_case(case, await asyncio.wrap_future(future), current_day)
        results[case.id] = result
        return {
            "answer": result.response.answer,
            "behavior": result.response.behavior.value,
            "intent": str(result.response.intent),
            "sources": result.response.sources,
            "checks": result.checks,
            "passed": result.passed,
            "source_recall": result.source_recall,
        }

    experiment = await asyncio.to_thread(
        client.run_experiment,
        name=dataset,
        run_name=run_name,
        data=[item for item in items if item.id in by_item],
        task=task,
        evaluators=[check_scores],
        run_evaluators=[pass_rate],
        max_concurrency=concurrency,
        metadata=metadata,
    )
    missing = [case.id for case in cases if case.id not in results]
    if missing:
        raise RuntimeError(f"cases without a result (see the log): {', '.join(missing)}")
    traces = {
        by_item[r.item.id].id: r.trace_id
        for r in experiment.item_results
        if isinstance(r.item, DatasetItem) and r.trace_id
    }
    return [results[case.id] for case in cases], traces, experiment.experiment_url


@contextmanager
def case_trace(client: Langfuse | None, trace_id: str | None, name: str) -> Iterator[None]:
    """Attach later work on a case (the judge) to the case's trace."""
    if client is None or trace_id is None:
        yield
        return
    with client.start_as_current_observation(
        as_type="span", name=name, trace_context={"trace_id": trace_id}
    ):
        yield


def record_groundedness(
    client: Langfuse, results: Sequence[CaseResult], traces: dict[str, str]
) -> int:
    """The judge's verdicts as scores on the cases' traces; returns how many were written."""
    written = 0
    for result in results:
        trace_id = traces.get(result.case_id)
        if result.grounded is None or trace_id is None:
            continue
        client.create_score(
            trace_id=trace_id,
            name=JUDGE_SCORE,
            value=1.0 if result.grounded else 0.0,
            data_type="BOOLEAN",
            comment="LLM judge judge/v1; uncalibrated until checked against human labels",
        )
        written += 1
    return written


def ensure_calibration_queue(client: Langfuse) -> str:
    """The score config for human groundedness labels and its annotation queue, created once;
    returns the queue id."""
    configs = client.api.score_configs.get(limit=100).data
    config = next((c for c in configs if c.name == HUMAN_SCORE and not c.is_archived), None)
    if config is None:
        config = client.api.score_configs.create(
            name=HUMAN_SCORE,
            data_type=ScoreConfigDataType("BOOLEAN"),
            description="Is every factual claim of the answer supported by the documents it "
            "cites? Human label that calibrates the LLM judge.",
        )
    queues = client.api.annotation_queues.list_queues(limit=100).data
    queue = next((q for q in queues if q.name == CALIBRATION_QUEUE), None)
    if queue is None:
        queue = client.api.annotation_queues.create_queue(
            name=CALIBRATION_QUEUE,
            score_config_ids=[config.id],
            description="Answers the LLM judge rated. Label them here, then export with "
            "customer-agent-eval --export-labels.",
        )
    return str(queue.id)


def queue_for_labelling(
    client: Langfuse, queue_id: str, results: Sequence[CaseResult], traces: dict[str, str]
) -> int:
    """Every judged case of the run goes to the queue; returns how many were added."""
    queued = 0
    for result in results:
        trace_id = traces.get(result.case_id)
        if result.grounded is None or trace_id is None:
            continue
        client.api.annotation_queues.create_queue_item(
            queue_id, object_id=trace_id, object_type=AnnotationQueueObjectType("TRACE")
        )
        queued += 1
    return queued


def _flag(value: object) -> bool:
    """Boolean scores come back as bool, 0/1 or 'True'/'False', depending on the endpoint."""
    return str(value).strip().lower() in ("true", "1", "1.0")


def _object(payload: object) -> dict[str, Any]:
    """The observations API returns input and output as raw JSON strings."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def labelled_rows(client: Langfuse, queue_id: str) -> list[dict[str, object]]:
    """The queue's human labels as calibration rows (the ``--check-calibration`` format), with
    the judge's verdict on the same trace; the latest label per case wins."""
    rows: dict[str, dict[str, object]] = {}
    cursor: str | None = None
    while True:
        page = client.api.scores.get_many(
            queue_id=queue_id, name=HUMAN_SCORE, fields="core,subject", limit=100, cursor=cursor
        )
        for score in sorted(page.data, key=lambda s: s.timestamp):
            subject = score.subject
            trace_id = getattr(subject, "trace_id", None) or subject.id
            root = client.api.observations.get_many(
                trace_id=trace_id, is_root_observation=True, fields="core,basic,io,metadata"
            ).data[0]
            judge = client.api.scores.get_many(
                trace_id=trace_id, name=JUDGE_SCORE, fields="core", limit=1
            ).data
            case_input = _object(root.input)
            output = _object(root.output)
            case_id = str((root.metadata or {}).get("case_id", trace_id))
            rows[case_id] = {
                "case_id": case_id,
                "question": case_input.get("turns", [{}])[-1].get("content", ""),
                "answer": output.get("answer", ""),
                "sources": output.get("sources", []),
                "judge_grounded": _flag(judge[0].value) if judge else None,
                "human_grounded": _flag(score.value),
                "trace_id": trace_id,
            }
        cursor = page.meta.cursor
        if not page.data or not cursor:
            break
    return [rows[case_id] for case_id in sorted(rows)]
