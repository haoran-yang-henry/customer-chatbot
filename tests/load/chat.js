// k6 load test (Day 7, NFR-09): 20 concurrent customers send knowledge questions to the API,
// which talks to the mock model (tests/load/mock_llm.py) instead of a real one.
// Default (NFR-09): 20 customers who read each answer for 2-6 s before writing again.
// THINK_MAX=0: no think time, every user sends the next message as soon as the answer arrives —
// a saturation test that measures capacity; latency there is mostly queueing, so the NFR-09
// threshold does not apply.
const THINK_MIN = Number(__ENV.THINK_MIN || 2);
const THINK_MAX = Number(__ENV.THINK_MAX || 6);
const SATURATION = THINK_MAX === 0;
import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8001";
const QUESTIONS = [
  "Wie lange habe ich Zeit, um einen Artikel zurückzuschicken?",
  "Kostet die Rücksendung etwas?",
  "Ab welchem Bestellwert ist der Versand kostenlos?",
  "Kann ich auf Rechnung bezahlen?",
  "Wie lange gilt bei euch die Gewährleistung?",
  "Liefert ihr auch in die Schweiz?",
];

export const options = {
  scenarios: {
    // Ramp up instead of starting every user in the same millisecond: a synchronized first
    // burst of 20 turns queues on one worker and dominated the p95 of the first runs.
    customers: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: __ENV.RAMP || "10s", target: Number(__ENV.VUS || 20) },
        { duration: __ENV.DURATION || "60s", target: Number(__ENV.VUS || 20) },
      ],
      gracefulRampDown: "5s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    // NFR-09 (model latency is zero); the saturation run only records the sub-metric
    "http_req_duration{endpoint:chat}": [SATURATION ? "p(95)<60000" : "p(95)<300"],
  },
  summaryTrendStats: ["avg", "med", "p(90)", "p(95)", "p(99)", "max"],
};

// The API may answer on 127.0.0.1 before it is reachable from this container (port forwarding
// of Docker Desktop on WSL): wait for it, or the first second floods the results with failures.
export function setup() {
  for (let attempt = 0; attempt < 60; attempt++) {
    if (http.get(`${BASE_URL}/health/live`, { tags: { endpoint: "setup" } }).status === 200) {
      return;
    }
    sleep(0.5);
  }
  throw new Error(`API not reachable at ${BASE_URL}`);
}

export default function () {
  const question = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const response = http.post(`${BASE_URL}/v1/chat`, JSON.stringify({ message: question }), {
    headers: { "Content-Type": "application/json" },
    tags: { endpoint: "chat" },
  });
  check(response, {
    "status is 200": (r) => r.status === 200,
    "full path answered": (r) => r.status === 200 && r.json("behavior") === "answer",
  });
  if (THINK_MAX > 0) {
    sleep(THINK_MIN + Math.random() * (THINK_MAX - THINK_MIN));
  }
}

// Machine-readable summary for tests/load/report.py, plus the key numbers on stdout.
export function handleSummary(data) {
  const chat = data.metrics["http_req_duration{endpoint:chat}"].values;
  const line = `chat p95=${chat["p(95)"].toFixed(1)}ms p99=${chat["p(99)"].toFixed(1)}ms ` +
    `requests=${data.metrics.http_reqs.values.count} failed=${data.metrics.http_req_failed.values.rate}\n`;
  return {
    stdout: line,
    [`${__ENV.OUT_DIR || "."}/k6-summary.json`]: JSON.stringify(data),
  };
}
