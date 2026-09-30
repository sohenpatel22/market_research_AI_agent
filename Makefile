.PHONY: sync lint format test db-up db-down ingest train retrieval-eval eval eval-gate serve docker-up docker-down docker-tools

sync:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

db-up:
	docker compose up -d

db-down:
	docker compose down

ingest:
	uv run python -m market_research_agent.data.ingest

train:
	uv run python -m market_research_agent.models.train

retrieval-eval:
	uv run python -m market_research_agent.eval.retrieval_eval

eval:
	uv run python -m market_research_agent.eval.run_eval --name dev

eval-gate:
	uv run pytest -m eval

serve:
	uv run python -m market_research_agent.api

docker-up:
	GIT_SHA=$$(git rev-parse --short HEAD) docker compose up -d --build

docker-down:
	docker compose down

docker-tools:
	docker compose --profile tools run --rm tools $(CMD)
