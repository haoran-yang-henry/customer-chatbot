.DEFAULT_GOAL := help
# Pinned by digest, like in .github/workflows/ci.yml (keep both in sync).
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f
TRIVY_IMAGE := aquasec/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969
.PHONY: seed token help install lint format typecheck test test-integration test-llm check audit eval eval-redteam eval-retrieval migrate ingest run up up-app up-observability down

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

eval:  ## End-to-end evaluation of the real agent + judge (needs make up migrate seed ingest)
	CA_LLM_CACHE_DIR=.cache/llm uv run customer-agent-eval --dataset eval/datasets/core.jsonl --day 6 \
		--target agent --judge --calibration-out eval/calibration/groundedness-latest.jsonl

eval-redteam:  ## Red-team set: attack success and false-positive rates (same setup as eval)
	CA_LLM_CACHE_DIR=.cache/llm uv run customer-agent-eval --dataset eval/datasets/redteam.jsonl \
		--day 6 --target agent

eval-retrieval:  ## Recall@5 / MRR for fulltext, vector and hybrid search (needs DB + API key)
	uv run customer-agent-eval-retrieval --dataset eval/datasets/core.jsonl --out eval/results/day2-retrieval.json

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

up:  ## Start local infrastructure (Postgres + pgvector, Keycloak)
	docker compose up -d --wait

up-app:  ## Build the API image and run it at http://localhost:8000 (after make up migrate seed ingest)
	docker compose --profile app up -d --build app

up-observability:  ## Jaeger for traces (UI http://localhost:16686); set CA_OTEL_ENDPOINT
	docker compose --profile observability up -d jaeger

down:  ## Stop local infrastructure
	docker compose --profile app --profile observability down
