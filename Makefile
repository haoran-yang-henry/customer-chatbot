.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration test-llm check eval eval-retrieval migrate ingest run up down

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

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

eval:  ## End-to-end evaluation of the real agent with the groundedness judge (cached LLM calls)
	CA_LLM_CACHE_DIR=.cache/llm uv run customer-agent-eval --dataset eval/datasets/core.jsonl --day 3 \
		--target agent --judge --calibration-out eval/calibration/groundedness-day3.jsonl

eval-retrieval:  ## Recall@5 / MRR for fulltext, vector and hybrid search (needs DB + API key)
	uv run customer-agent-eval-retrieval --dataset eval/datasets/core.jsonl --out eval/results/day2-retrieval.json

migrate:  ## Apply database migrations
	uv run alembic upgrade head

ingest:  ## Synchronise the knowledge index with data/knowledge (needs DB + API key)
	uv run customer-agent-ingest

run:  ## Start the API with auto-reload; live logs in this terminal (needs DB + both API keys)
	uv run uvicorn --factory customer_agent.main:create_app --reload --no-access-log

up:  ## Start local infrastructure (Postgres + pgvector)
	docker compose up -d --wait

down:  ## Stop local infrastructure
	docker compose down
