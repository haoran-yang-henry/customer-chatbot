"""Combines the k6 summary with the server's own step histograms into one result file.

    python -m tests.load.report --metrics URL --k6 summary.json --out eval/results/day7-load.json

Server-side percentiles are bucket upper bounds (Prometheus histograms); means are exact.
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from prometheus_client.parser import text_string_to_metric_families


def histograms(text: str, name: str, label: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        if family.name != name:
            continue
        for sample in family.samples:
            if sample.labels.get("outcome", "ok") != "ok":
                continue
            key = sample.labels.get(label, "")
            if sample.name.endswith("_bucket"):
                buckets[key].append((float(sample.labels["le"]), sample.value))
            elif sample.name.endswith("_sum"):
                sums[key] = sample.value
            elif sample.name.endswith("_count"):
                counts[key] = sample.value
    result = {}
    for key, count in counts.items():
        if not count:
            continue
        ordered = sorted(buckets[key])
        p95 = next(le for le, cumulative in ordered if cumulative >= 0.95 * count)
        within_250ms = next(cumulative for le, cumulative in ordered if le >= 0.25)
        result[key] = {
            "count": int(count),
            "mean_ms": round(sums[key] / count * 1000, 1),
            "p95_ms_at_most": None if p95 == float("inf") else round(p95 * 1000),
            "over_250ms": int(count - within_250ms),
        }
    return result


def overhead(steps: dict[str, dict[str, Any]], http: dict[str, dict[str, Any]]) -> float | None:
    """Mean time per chat request not spent waiting for a model: the request minus the router
    call (the classifier runs alongside it), the query embedding and the streamed answer."""
    chat = http.get("/v1/chat")
    if chat is None:
        return None
    turns = float(chat["count"])

    def per_turn(step: str) -> float:
        data = steps.get(step)
        return float(data["mean_ms"]) * float(data["count"]) / turns if data else 0.0

    model = per_turn("route") + per_turn("embed") + per_turn("llm_stream")
    return round(float(chat["mean_ms"]) - model, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--k6", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock-latency-ms", type=float, default=0.0)
    parser.add_argument("--scenario", default="")
    args = parser.parse_args()

    with urllib.request.urlopen(args.metrics, timeout=10) as response:  # noqa: S310 - local URL
        text = response.read().decode()
    k6 = json.loads(args.k6.read_text(encoding="utf-8"))["metrics"]
    duration = k6["http_req_duration{endpoint:chat}"]["values"]
    report = {
        "nfr": "NFR-09: P95 <= 300 ms system overhead at 20 concurrent users (model latency 0)",
        "mock_llm_latency_ms": args.mock_latency_ms,
        "client": {
            "requests": k6["http_reqs"]["values"]["count"],
            "throughput_rps": round(k6["http_reqs"]["values"]["rate"], 1),
            "failed_rate": k6["http_req_failed"]["values"]["rate"],
            "full_path_checks": k6["checks"]["values"]["rate"],
            "latency_ms": {k: round(v, 1) for k, v in duration.items()},
        },
        "server_steps": histograms(text, "ca_step_duration_seconds", "step"),
        "server_http": histograms(text, "ca_http_request_duration_seconds", "route"),
    }
    report["scenario"] = args.scenario
    report["overhead_mean_ms"] = overhead(report["server_steps"], report["server_http"])
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["client"], indent=2))


if __name__ == "__main__":
    main()
