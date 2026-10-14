#!/usr/bin/env bash
# Smoke evaluation with recorded model responses: the CI quality gate.
#
#   eval/smoke.sh           replay: offline and deterministic, no API keys (what CI runs)
#   eval/smoke.sh record    record: real providers (through the local LLM cache), rewrites
#                           eval/cassettes/smoke.json.gz — commit it with the change that needed it
#
# A fresh database every time: chunk ids break ties in retrieval, so the evidence order and
# therefore every prompt match the recording only for an identical, freshly ingested index.
# Locally the database runs in docker compose (`make up`); CI sets SMOKE_DATABASE_URL.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-replay}
CASSETTE=eval/cassettes/smoke.json.gz
DB=customer_agent_smoke

export CA_DATABASE_URL=${SMOKE_DATABASE_URL:-postgresql+psycopg://agent:agent@localhost:5432/$DB}
export CA_EVAL_CASSETTE=$CASSETTE CA_LOG_LEVEL=WARNING
case "$MODE" in
  record)
    export CA_EVAL_CASSETTE_RECORD=true CA_LLM_CACHE_DIR=.cache/llm
    rm -f "$CASSETTE"  # one consistent run, no stale entries
    ;;
  replay)
    # Never used: in replay mode no provider client is built.
    export CA_EVAL_CASSETTE_RECORD=false CA_LLM_CACHE_DIR= CA_OPENAI_API_KEY=unused CA_LLM_API_KEY=unused
    ;;
  *)
    echo "usage: $0 [replay|record]" >&2
    exit 2
    ;;
esac

if [ -z "${SMOKE_DATABASE_URL:-}" ]; then
  docker compose exec -T postgres dropdb -U agent --if-exists "$DB"
  docker compose exec -T postgres createdb -U agent "$DB"
fi
uv run alembic upgrade head
uv run customer-agent-seed > /dev/null
uv run customer-agent-ingest > /dev/null
uv run customer-agent-eval --target agent --day 8 \
  --dataset eval/datasets/core.jsonl --dataset eval/datasets/redteam.jsonl --tag smoke \
  --min-pass-rate 1 --out "${SMOKE_OUT:-eval/results/smoke-$MODE.json}"
