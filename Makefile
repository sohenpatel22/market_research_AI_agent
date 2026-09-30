.PHONY: sync lint format test db-up db-down ingest train

sync:
	uv sync --extra dev

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
