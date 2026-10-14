#!/usr/bin/env bash
# Load test (Day 7, NFR-09): the API against the mock model, driven by k6.
# Needs `make up` (Postgres) and Docker. Uses its own database, so the development index and
# its real embeddings stay untouched; the knowledge base is re-ingested with mock embeddings.
#
#   tests/load/run.sh [result file]
#   DURATION=60s VUS=20 THINK_MIN=2 THINK_MAX=6 MOCK_LLM_LATENCY_MS=0 CA_DB_POOL_SIZE=20
#   THINK_MAX=0: saturation (capacity) instead of the NFR-09 scenario
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:-eval/results/day7-load.json}
MOCK_PORT=9100
API_PORT=8001
DB=customer_agent_load
K6_IMAGE="grafana/k6:2.3.0@sha256:9c2dee7f8ed74d317e4027c06a10f169b625638189de8d4555d0b3486a5aeb34"
WORK=$(mktemp -d)

# Environment variables override .env (pydantic-settings); empty values switch features off.
export CA_DATABASE_URL="postgresql+psycopg://agent:agent@localhost:5432/$DB"
export CA_OPENAI_API_KEY=mock CA_OPENAI_BASE_URL="http://127.0.0.1:$MOCK_PORT/v1"
export CA_LLM_API_KEY=mock CA_LLM_BASE_URL="http://127.0.0.1:$MOCK_PORT/v1"
export CA_LLM_CACHE_DIR= CA_LLM_SESSION_HEADER= CA_LLM_REASONING_EFFORT=
export CA_AUTH_ISSUER= CA_RATE_LIMIT_PER_MINUTE=0 CA_LOG_LEVEL=WARNING
export CA_OTEL_ENDPOINT=${LOAD_OTEL_ENDPOINT:-}  # e.g. http://localhost:4318 to inspect slow traces
export MOCK_LLM_LATENCY_MS=${MOCK_LLM_LATENCY_MS:-0}

wait_for() {
  for _ in $(seq 1 60); do
    curl -fsS -o /dev/null "$1" && return 0
    sleep 0.5
  done
  echo "timed out waiting for $1" >&2
  return 1
}

cleanup() {
  kill "${API_PID:-}" "${MOCK_PID:-}" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

if ! docker compose exec -T postgres psql -U agent -d customer_agent -Atc \
  "SELECT 1 FROM pg_database WHERE datname = '$DB'" | grep -q 1; then
  docker compose exec -T postgres createdb -U agent "$DB"
fi

uv run uvicorn tests.load.mock_llm:app --port "$MOCK_PORT" --log-level warning \
  --timeout-keep-alive 75 &
MOCK_PID=$!
wait_for "http://127.0.0.1:$MOCK_PORT/docs"

uv run alembic upgrade head
uv run customer-agent-ingest > /dev/null

uv run uvicorn --factory customer_agent.main:create_app --host 0.0.0.0 --port "$API_PORT" \
  --no-access-log --log-level warning --timeout-keep-alive 75 &
API_PID=$!
wait_for "http://127.0.0.1:$API_PORT/health/ready"

# k6 runs in a container and reaches the API on the host through host.docker.internal
# (Docker Desktop and Linux alike); `--network host` does not reach a WSL distro's localhost.
K6_STATUS=0
docker run --rm --add-host=host.docker.internal:host-gateway --user "$(id -u):$(id -g)" \
  -v "$PWD/tests/load:/scripts:ro" -v "$WORK:/out" \
  -e BASE_URL="http://host.docker.internal:$API_PORT" -e OUT_DIR=/out \
  -e DURATION="${DURATION:-60s}" -e VUS="${VUS:-20}" \
  -e THINK_MIN="${THINK_MIN:-2}" -e THINK_MAX="${THINK_MAX:-6}" \
  "$K6_IMAGE" run --quiet /scripts/chat.js || K6_STATUS=$?

uv run python -m tests.load.report --metrics "http://127.0.0.1:$API_PORT/metrics" \
  --k6 "$WORK/k6-summary.json" --out "$OUT" --mock-latency-ms "$MOCK_LLM_LATENCY_MS" \
  --scenario "vus=${VUS:-20} think=${THINK_MIN:-2}-${THINK_MAX:-6}s db_pool=${CA_DB_POOL_SIZE:-20}"
echo "written to $OUT"
if [ "$K6_STATUS" -ne 0 ]; then
  echo "k6 thresholds failed (exit $K6_STATUS)" >&2
  exit "$K6_STATUS"
fi
