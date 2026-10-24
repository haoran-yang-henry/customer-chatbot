.DEFAULT_GOAL := help
# Pinned by digest, like in .github/workflows/ci.yml (keep both in sync).
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f
TRIVY_IMAGE := aquasec/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969
.PHONY: seed token help install lint format typecheck test test-integration test-llm check deps-check audit eval eval-labels eval-smoke eval-smoke-record eval-redteam eval-retrieval ttft load-test load-test-capacity migrate ingest run up up-app up-observability down demo-public demo-public-down demo-frontend

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Static type check (mypy strict)
	uv run mypy

test:  ## Run unit and data tests with coverage
	uv run pytest --cov --cov-report=term-missing -m "not integration and not llm"

test-integration:  ## Integration tests, hashing embedder (throwaway Postgres via Testcontainers)
	uv run pytest -m integration

test-llm:  ## All integration tests with the real embedding model + production scenarios (API key)
	uv run pytest -m integration --embeddings=openai

check: lint typecheck test  ## Fast checks (no Docker, no API keys)

deps-check:  ## Unused or undeclared packages (deptry) and dead code (vulture, see vulture_whitelist.py)
	uv run deptry src
	uv run vulture

audit:  ## Security scans: secrets in git history, Trivy on the tree, pip-audit (needs Docker)
	docker run --rm -v "$(CURDIR):/repo:ro" --user "$$(id -u):$$(id -g)" $(GITLEAKS_IMAGE) \
		git /repo --redact --no-banner
	docker run --rm -v "$(CURDIR):/repo:ro" -v trivy-cache:/root/.cache $(TRIVY_IMAGE) \
		fs --scanners vuln,secret,misconfig --severity HIGH,CRITICAL --exit-code 1 \
		--skip-version-check --skip-dirs .venv --skip-dirs .cache --skip-files .env /repo
	uv export --locked --quiet --format requirements-txt --no-emit-project --all-groups \
		-o .cache/requirements-audit.txt
	uvx --from pip-audit==2.10.1 pip-audit -r .cache/requirements-audit.txt --require-hashes \
		--disable-pip

# LANGFUSE=1 records eval and eval-redteam as Langfuse experiments and queues judged answers for
# human labels (needs the LANGFUSE_* keys in .env).
LANGFUSE_FLAGS := $(if $(LANGFUSE),--langfuse,)

eval:  ## End-to-end evaluation of the real agent + judge (needs make up migrate seed ingest)
	CA_LLM_CACHE_DIR=.cache/llm uv run customer-agent-eval --dataset eval/datasets/core.jsonl --day 8 \
		--target agent --judge --calibration-out eval/calibration/groundedness-latest.jsonl \
		$(LANGFUSE_FLAGS) $(if $(LANGFUSE),--annotation-queue,)

eval-labels:  ## Export the human labels from Langfuse's calibration queue and check the judge
	uv run customer-agent-eval --export-labels eval/calibration/groundedness-langfuse.jsonl

eval-smoke:  ## CI gate locally: smoke cases against recorded model responses (needs make up)
	eval/smoke.sh

eval-smoke-record:  ## Re-record the smoke cassette with the real providers (after prompt changes)
	eval/smoke.sh record

eval-redteam:  ## Red-team set: attack success and false-positive rates (same setup as eval)
	CA_LLM_CACHE_DIR=.cache/llm uv run customer-agent-eval --dataset eval/datasets/redteam.jsonl \
		--day 8 --target agent $(LANGFUSE_FLAGS)

eval-retrieval:  ## Recall@5 / MRR for fulltext, vector and hybrid search (needs DB + API key)
	uv run customer-agent-eval-retrieval --dataset eval/datasets/core.jsonl --out eval/results/day2-retrieval.json

ttft:  ## Time to first token with the real model, NFR-10 (needs: CA_RATE_LIMIT_PER_MINUTE=0 make run)
	uv run python -m tests.load.ttft --out eval/results/day8-ttft.json

load-test:  ## NFR-09: 20 customers with think time, mocked model (needs make up and Docker)
	tests/load/run.sh

load-test-capacity:  ## Saturation: 20 users without think time, measures turns per second
	THINK_MIN=0 THINK_MAX=0 tests/load/run.sh eval/results/day7-load-capacity.json

migrate:  ## Apply database migrations
	uv run alembic upgrade head

seed:  ## Load data/seed into the mock order system (idempotent)
	uv run customer-agent-seed

ingest:  ## Synchronise the knowledge index with data/knowledge (needs DB + API key)
	uv run customer-agent-ingest

run:  ## Start the API with auto-reload; live logs in this terminal (needs DB + both API keys)
	uv run uvicorn --factory customer_agent.main:create_app --reload --no-access-log

token:  ## Print an access token from the local Keycloak, e.g. make token USER=anna.schneider
	@uv run customer-agent-token $(USER)

demo-public:  ## Browser demo: frontend on GitHub Pages, this machine as backend via a Cloudflare tunnel
	infra/demo/public.sh

demo-public-down:  ## Stop the browser demo's tunnel and restore the local setup
	infra/demo/public.sh down

demo-frontend:  ## Serve the browser demo locally on http://localhost:3000 (allowed by Keycloak and CORS)
	python3 -m http.server 3000 --bind 127.0.0.1 -d frontend/public

up:  ## Start local infrastructure (Postgres + pgvector, Keycloak)
	docker compose up -d --wait

up-app:  ## Build the API image and run it at http://localhost:8000 (after make up migrate seed ingest)
	docker compose --profile app up -d --build app

up-observability:  ## Jaeger for traces (UI http://localhost:16686); set CA_OTEL_ENDPOINT
	docker compose --profile observability up -d jaeger

down:  ## Stop local infrastructure
	docker compose --profile app --profile observability down
