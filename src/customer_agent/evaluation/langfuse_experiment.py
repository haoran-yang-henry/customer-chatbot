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
"""

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import Evaluation, ExperimentItem, ExperimentItemResult
from opentelemetry import context as otel_context

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import AgentResponse
from customer_agent.evaluation.schema import EvalCase
from customer_agent.evaluation.scoring import CaseResult, score_case

# Langfuse's generated enums type their members as plain str; looked up by value they are typed.
ACTIVE = DatasetStatus("ACTIVE")
ARCHIVED = DatasetStatus("ARCHIVED")


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
            name="grounded",
            value=1.0 if result.grounded else 0.0,
            data_type="BOOLEAN",
            comment="LLM judge judge/v1; uncalibrated until checked against human labels",
        )
        written += 1
    return written
