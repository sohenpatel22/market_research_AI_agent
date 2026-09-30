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
make db-up   # start Postgres (localhost:5433)
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

## Forecasting models (Phase 2)

```bash
uv run python -m market_research_agent.models.train        # trains, evaluates, logs to MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs and the model registry
```

Code lives in `src/market_research_agent/models/`. Everything is evaluated on a chronological
70/15/15 train/val/test split with purged boundaries (no shuffling across time).

* **Volatility (next 5 trading days)**: persistence, HAR-RV, GARCH(1,1)-t and ARIMA baselines vs a
  pooled PyTorch LSTM. The daily variance proxy is the Parkinson high-low estimator. Metrics: RMSE, MAE,
  QLIKE, plus a Diebold-Mariano test (HAC) against HAR.
* **Direction (next ~21 trading days)**: scikit-learn logistic regression / gradient boosting, picked
  on validation ROC-AUC; reports ROC-AUC, F1 and a confusion matrix against the base rate.
* **Inference**: `forecast(ticker, horizon="1w"|"1m") -> ForecastResult` (Pydantic), loading the newest
  bundle from `artifacts/models/`. This is what the agent's forecast tool will call.

Latest run (5 tickers, ~10y daily data, test set = most recent 15%):

| Model | RMSE (vol) | QLIKE |
|---|---|---|
| LSTM | 0.0840 | 0.202 |
| HAR-RV | 0.0890 | 0.223 |
| ARIMA | 0.0885 | 0.239 |
| Persistence | 0.1062 | 0.288 |
| GARCH | 0.0991 | 0.311 |

The LSTM beats HAR (Diebold-Mariano p = 0.001). The direction classifier is weak
(test ROC-AUC 0.62, accuracy below the 65% always-up base rate), which is the honest expectation for
return prediction. Outputs are statistical estimates, not investment advice.
