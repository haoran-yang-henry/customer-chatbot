"""Time to first token as the customer sees it (NFR-10), against a running API and the real model.

    CA_RATE_LIMIT_PER_MINUTE=0 CA_LLM_CACHE_DIR= make run      # in another shell
    uv run python -m tests.load.ttft --out eval/results/day8-ttft.json

Sends the single-turn knowledge questions of the evaluation datasets one after the other to
``/v1/chat/stream`` and measures, per turn, the time until the first visible text (the first
``delta`` event, or the ``final`` event when nothing is streamed) and until the ``final`` event.
Requests are sequential on purpose: this measures the model provider's latency through the whole
pipeline, not the API's capacity (that is the k6 load test).
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx2

from customer_agent.domain.conversation import Behavior
from customer_agent.evaluation.schema import load_dataset

DATASETS = [Path("eval/datasets/core.jsonl"), Path("eval/datasets/retrieval_holdout.jsonl")]
TARGET_SECONDS = 3.0


def questions(limit: int | None) -> list[tuple[str, str]]:
    cases = [case for path in DATASETS for case in load_dataset(path)]
    selected = [
        (case.id, case.turns[0].content)
        for case in cases
        if len(case.turns) == 1
        and case.customer_id is None
        and case.expected_behavior is Behavior.ANSWER
    ]
    return selected[:limit]


def measure(client: httpx2.Client, url: str, message: str) -> dict[str, Any]:
    start = time.perf_counter()
    first_text: float | None = None
    behavior = None
    with client.stream("POST", url, json={"message": message}) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event in ("delta", "final"):
                if first_text is None:
                    first_text = time.perf_counter() - start
                if event == "final":
                    behavior = json.loads(line.removeprefix("data: "))["behavior"]
    total = time.perf_counter() - start
    return {
        "first_text_s": round(first_text or total, 3),
        "total_s": round(total, 3),
        "behavior": behavior,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]  # nearest rank


def summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    turns = []
    with httpx2.Client(timeout=60) as client:
        for case_id, message in questions(args.limit):
            result = {
                "case": case_id,
                **measure(client, f"{args.base_url}/v1/chat/stream", message),
            }
            print(
                f"{case_id}  first text {result['first_text_s']:6.2f} s  "
                f"final {result['total_s']:6.2f} s  {result['behavior']}"
            )
            turns.append(result)

    first_text = summary([t["first_text_s"] for t in turns])
    report = {
        "requirement": "NFR-10: time to first token p95 <= 3 s",
        "turns": len(turns),
        "first_text_s": first_text,
        "total_s": summary([t["total_s"] for t in turns]),
        "met": first_text["p95"] <= TARGET_SECONDS,
        "by_turn": turns,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "by_turn"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
