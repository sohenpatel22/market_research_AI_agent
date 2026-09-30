# Market Research Agent

An agentic RAG assistant over SEC filings and stock price data that can also call a forecasting model.

## Status

**Phase 1 — data layer.** Postgres + pgvector, price/fundamentals/filing-chunk ingestion, and
DVC-versioned raw snapshots are in place. No agent or forecasting logic yet.

## Project layout

```
src/market_research_agent/
    data/       # Data ingestion, loading, and preprocessing (SEC filings, price data)
    models/     # Forecasting model training/inference code
    agent/      # Agentic RAG orchestration (retrieval, tool-calling, planning)
    api/        # API layer exposing the assistant (e.g. FastAPI app)
    eval/       # Evaluation harnesses and metrics for retrieval/forecasting/agent quality
    viz/        # Plotting and reporting utilities
tests/          # Unit and integration tests (mirrors src/ layout)
notebooks/      # Exploratory analysis, not imported by application code
data/           # Local data files: DVC-tracked, not git-tracked (see Data layer below)
.github/workflows/  # CI pipelines
docker-compose.yml  # Local Postgres 16 + pgvector
dvc-storage/        # Local DVC remote (git-ignored)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and pins Python 3.11.

```bash
uv sync --extra dev
cp .env.example .env  # then fill in real values
```

## Commands

| Task        | Make (Linux/macOS/CI) | PowerShell (Windows)     |
|-------------|------------------------|---------------------------|
| Lint        | `make lint`            | `./scripts/lint.ps1`      |
| Format      | `make format`          | `./scripts/format.ps1`    |
| Test        | `make test`            | `./scripts/test.ps1`      |
| Start DB    | `make db-up`           | `./scripts/db-up.ps1`     |
| Stop DB     | `make db-down`         | `./scripts/db-down.ps1`   |
| Ingest data | `make ingest`          | `./scripts/ingest.ps1`    |

## Configuration

Settings are loaded from environment variables (or a local `.env` file) via `pydantic-settings` in
[`src/market_research_agent/config.py`](src/market_research_agent/config.py). See `.env.example` for the required
variables. Never commit `.env`.

## Data layer

### Database

`docker-compose.yml` runs Postgres 16 with the [pgvector](https://github.com/pgvector/pgvector)
extension enabled, matching the default `DATABASE_URL` in `.env.example`.

```bash
make db-up   # start Postgres (localhost:5432)
uv run python -m market_research_agent.data.ingest  # creates the schema, then ingests
make db-down # stop Postgres
```

Schema (see [`src/market_research_agent/data/models.py`](src/market_research_agent/data/models.py)):

- **prices** — daily OHLCV bars, one row per `(ticker, date)`.
- **fundamentals** — quarterly financial line items, stored long/tidy as one row per
  `(ticker, period, metric)` rather than wide, since the set of reported metrics varies by company.
- **documents** — chunked 10-K/10-Q filing text with a pgvector `embedding` column (384-dim,
  matching the default `BAAI/bge-small-en-v1.5` sentence-transformer).

### Ingestion

`uv run python -m market_research_agent.data.ingest [--tickers AAPL MSFT ...]` (defaults to
AAPL, MSFT, NVDA, JPM, XOM) does the following per ticker:

1. Downloads daily price history and quarterly fundamentals via `yfinance`.
2. Downloads the most recent 10-K and 10-Q from SEC EDGAR (a descriptive `User-Agent` from
   `SEC_EDGAR_USER_AGENT` is required on every request, and requests are throttled to stay well
   under SEC's fair-access rate limit).
3. Cleans and chunks filing text, embeds each chunk locally with a Hugging Face
   sentence-transformer (no external embedding API), and upserts everything into Postgres.

Every raw response (price CSVs, fundamentals CSVs, filing HTML) is cached under `data/raw/` on
first fetch, so re-running ingestion is cheap and doesn't re-hit yfinance/EDGAR. All inserts are
idempotent (`ON CONFLICT DO NOTHING` on the natural key), so re-running ingestion is also safe
against the database.

### Versioning raw data with DVC

`data/raw/` is tracked with [DVC](https://dvc.org/), not git — git only tracks the small
`data/raw.dvc` pointer file. A local DVC remote lives at `dvc-storage/` (git-ignored) for now.

```bash
# after ingestion has (re)populated data/raw/:
uv run dvc add data/raw       # updates data/raw.dvc with the new content hash
git add data/raw.dvc
uv run dvc push               # uploads the data to the local remote (dvc-storage/)

# on another machine / after a fresh clone:
uv run dvc pull                # restores data/raw/ from the local remote
```
