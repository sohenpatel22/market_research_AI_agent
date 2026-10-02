# Market Research Agent

[![CI](https://github.com/sohenpatel22/market_research_AI_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sohenpatel22/market_research_AI_agent/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/SohenP/market-research-agent)

An agentic RAG assistant over SEC filings and stock prices. It routes a plain-English question,
retrieves filing excerpts (hybrid vector + keyword search with a cross-encoder reranker), calls
trained volatility/direction forecasters and whitelisted SQL lookups, drafts a cited answer, and has
a judge model verify it against the sources, retrying with a rewritten search when it is weak.

**Try it:** <https://huggingface.co/spaces/SohenP/market-research-agent> (free tier: the first
request after a quiet spell can take a minute while the Space wakes up).

| Area | What is in the repo |
|---|---|
| Agent | LangGraph route / gather / generate / grade / rewrite loop, provider-agnostic LLM (DeepSeek, OpenAI, Anthropic), prompt-injection guardrails |
| Retrieval | pgvector + Postgres full-text fused with RRF, section-aware 10-K/10-Q chunking, cross-encoder rerank |
| Models | PyTorch LSTM vs HAR-RV / GARCH / ARIMA volatility forecasts, scikit-learn direction classifier, MLflow tracking |
| Evaluation | 38-question golden set, RAGAS dev loop, DeepEval CI gate, retrieval ablation, cost/latency tracking |
| Observability | Langfuse traces, scores, prompt registry and per-model cost |
| Serving | FastAPI (+ SSE streaming), Gradio UI, rate limiting, Docker, GitHub Actions CI/CD to a Hugging Face Space |

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
uv sync --all-extras
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

## Agent (Phase 3)

`src/market_research_agent/agent/` is a LangGraph state machine:
`route -> gather -> generate -> grade -> (rewrite -> gather ...) -> finalize`.

* **route**: the LLM classifies the question and picks tools; out-of-scope requests (trades,
  personalized advice, prompt-override attempts) are refused without touching any tool.
* **gather**: hybrid filing search (Postgres full-text + pgvector fused with reciprocal rank
  fusion, optional cross-encoder reranking via `USE_RERANKER=true`), the forecasting models, and
  whitelisted read-only SQL lookups. Filings are chunked by 10-K/10-Q "Item" section.
* **grade / rewrite**: an LLM judge scores grounding and relevance; failures rewrite the search
  query and retry, bounded by `AGENT_MAX_RETRIES` (default 2) and a LangGraph recursion limit.
* **finalize**: a Pydantic `AgentAnswer` with citations verified against what was retrieved.
* Filing excerpts are treated as untrusted data (injection-looking lines are stripped, excerpts
  are wrapped in tags the system prompt marks as data).

```bash
uv run python -m market_research_agent.agent.cli "What supply chain risks does Apple report?"
```

The provider is chosen with `LLM_PROVIDER` (`deepseek` | `openai` | `anthropic`); see `.env.example`.

## Observability and cost control (Phase 4)

Use `agent/service.py::ask()` (the API, CLI and evals all go through it). With Langfuse configured
each question is one trace: route, tool calls and every LLM call with tokens, cost and latency,
tagged with provider, model and git sha, plus scores (`quality_passed`, `grade_score`, `retries`,
`refused`, `tool_errors`). Without keys it is a no-op, so tests, CI and forks need no Langfuse.

1. Create a free project at <https://cloud.langfuse.com> and set `LANGFUSE_PUBLIC_KEY` and
   `LANGFUSE_SECRET_KEY` in `.env`.
2. `uv run python -m market_research_agent.agent.cli "your question"` prints the answer and a trace link.
3. Optional prompt registry: `... agent.cli --sync-prompts` uploads the local prompts; then set
   `LANGFUSE_PROMPTS=true` to load the `production` versions (falls back to local text on any
   error or if a registry edit changes the template variables).

Keeping LLM spend low:

* one cheap model for both agent and judge by default (DeepSeek); a typical full question is
  about 5k tokens, and a refusal costs about 1k;
* `LLM_CACHE=true` caches identical calls on disk, so re-running evals/dev questions is free;
* the grade/retry loop is capped (`AGENT_MAX_RETRIES=2`), and forecasts/SQL lookups run once per
  question, not once per retry;
* structured-output calls retry (with a changed prompt) if the model skips the function call.

## Evaluation (Phase 5)

`eval/golden_dataset.json` holds 38 questions: 23 filing questions (drafted by an LLM from sampled
filing chunks so the ground-truth source is known, then hand-curated), 5 forecast, 5 price/fundamentals
lookups, and 5 out-of-scope/adversarial requests that must be refused.

| Command | What it does | Cost |
|---|---|---|
| `uv run python -m market_research_agent.eval.retrieval_eval` | dense vs sparse vs hybrid vs +rerank: NDCG@6, recall@6, MRR | free |
| `uv run python -m market_research_agent.eval.run_eval --name <run>` | runs the agent over the golden set; deterministic checks + RAGAS; logs to Langfuse | about $0.15 on DeepSeek |
| `uv run python -m market_research_agent.eval.compare` | combines saved runs into a comparison table | free |
| `uv run pytest -m eval` | the DeepEval CI gate on 6 items (`tests/eval/`) | about $0.03 |

Thresholds for all of these live in `eval/thresholds.yaml`. RAGAS is run deliberately during
development; the DeepEval gate runs in CI (`.github/workflows/ci.yml`, job `eval-gate`) against a small
committed corpus (`tests/eval/fixtures/ci_corpus.json`) so it works on an empty database, and
skips itself when no API key is configured. Judges are provider-agnostic (`JUDGE_PROVIDER`).

Baseline (DeepSeek, reranker on, cache off, n=38): faithfulness 0.95, answer relevancy 0.88,
context precision 0.80, context recall 0.96, refusal accuracy 100%, forecast/data tool use 100%,
about $0.0011 per question and 4.5k tokens, latency p50 4.5 s / p95 14 s (judge cost about $0.07
per full run).

Retrieval ablation (23 filing questions, k=6):

| Retriever | NDCG@6 | Recall@6 | MRR |
|---|---|---|---|
| dense | 0.483 | 0.826 | 0.605 |
| sparse (keywords) | 0.250 | 0.522 | 0.333 |
| hybrid (RRF) | 0.447 | 0.783 | 0.570 |
| hybrid + cross-encoder rerank | 0.591 | 0.913 | 0.772 |

Findings: the keyword leg alone is weak here, so plain RRF slightly trails dense-only; the
cross-encoder rerank gives the clear win, so it is on by default (`USE_RERANKER`). Down-weighting the
keyword leg helps the un-reranked hybrid but not the reranked one. With 23 questions these
differences are indicative, not statistically established. Only DeepSeek has been run end to end so
far; use `--provider`/`--model` to add OpenAI or Claude rows to the comparison.

## API and UI (Phase 6)

```bash
uv run python -m market_research_agent.api      # http://localhost:7860 (UI at /, docs at /docs)
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Ask a question; returns a validated `AgentAnswer` (answer, citations, forecasts, data, quality verdict) plus the Langfuse trace URL |
| `POST /chat/stream` | Same, as Server-Sent Events: `step` events (`route`, `gather`, `generate`, `grade`, `rewrite`) then one `final` event, or `error` |
| `POST /forecast` | `{"ticker": "AAPL", "horizon": "1w" \| "1m"}` runs the trained models |
| `GET /tickers` | Supported tickers and the model version |
| `GET /health` | Database, forecast models, LLM key and Langfuse status (`ok` / `degraded`) |

```bash
curl -N -X POST localhost:7860/chat/stream -H 'content-type: application/json' \
  -d '{"question": "What does NVIDIA say about export controls?"}'
```

Design notes:

* All bodies are Pydantic models with validation (question length, ticker pattern, horizon enum).
  Blocking work runs in a thread pool, so the event loop is never blocked; streamed requests run
  entirely on one worker thread because the Langfuse tracing context is thread-bound.
* The agent runtime is built lazily and without a checkpointer, so `/health` and `/forecast` work
  with no LLM key and requests don't accumulate state. A missing key is a clean `503`, never a stack trace.
* `RATE_LIMIT_PER_MINUTE` (default 30 per client IP) protects the paid LLM on `/chat` and `/forecast`
  (and the UI); errors returned to clients never include internal details.
* The Gradio UI (`viz/`) is a thin layer over the same `ask_stream`, `forecast` and price tables:
  a live progress checklist and cited sources in **Ask the agent**, a Plotly forecast chart in
  **Forecast**, and Matplotlib/Seaborn charts (relative performance, return distribution, correlation)
  in **Market data**.

## Docker (Phase 7)

```bash
make docker-up                  # builds the image, starts Postgres + the app on http://localhost:7860
docker compose logs -f app
make docker-down
```

* **`Dockerfile`** is multi-stage: dependencies are installed from `uv.lock` (cached layer) into a
  venv that is copied into a slim runtime image with **no compilers, no uv, and no
  training/eval/ingestion libraries** (those live in extras: `ingest`, `train`, `eval`). PyTorch is the CPU
  wheel, and the embedding and reranker models are downloaded at build time, so the container starts
  offline (`HF_HUB_OFFLINE=1`). It runs as a **non-root user (UID 1000)** and has a `HEALTHCHECK` on
  `/health`. Image size is about 3 GB uncompressed (PyTorch alone is 0.8 GB).
* **Secrets are never baked in.** `docker-compose.yml` reads your git-ignored `.env` at run time; inside
  the compose network `DATABASE_URL` is pointed at the `postgres` service automatically.
* **`docker-compose.yml`** starts `postgres` (pgvector), waits for it to be healthy, then the `app`.
  A `tools` profile provides the heavier libraries for one-off jobs against the same database:

```bash
docker compose --profile tools run --rm tools                                   # ingest prices/filings
docker compose --profile tools run --rm tools python -m market_research_agent.models.train
docker compose --profile tools run --rm tools python -m market_research_agent.eval.retrieval_eval
```

CI (`docker-build` job) builds the runtime image, checks it runs as UID 1000 with no secrets in its
layers, then starts the container against a Postgres service and asserts `/health`, the UI and request
validation respond.

## Deployment (Phase 8)

Production runs on a free **Hugging Face Docker Space** (the image from the `Dockerfile`) talking to a
free **Neon** Postgres with pgvector. A Space has no persistent database of its own, which is why the
corpus lives in Neon.

```
GitHub (master) --CI passes--> deploy.yml --> HF Space (builds the Docker image, runs on :7860)
                                                  |--> Neon Postgres (pgvector) : filings, prices
                                                  |--> DeepSeek API             : the language model
                                                  `--> Langfuse                 : traces and scores
```

**Workflows** (`.github/workflows/`):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | lint, tests, DeepEval gate, Docker build + container smoke test |
| `deploy.yml` | CI green on `master`, or manual | copies secrets into the Space, uploads a minimal build context, waits for the build, smoke-tests `/health`, the UI and one real `/chat` |
| `seed-database.yml` | manual | runs the ingestion pipeline against the production database (idempotent) |
| `preflight.yml` | manual | checks the database, pgvector and the Hugging Face token without printing secrets |

**Secrets** (GitHub repo settings; the deploy copies the runtime ones into the Space's own secrets):
`HF_TOKEN` (fine-grained, write access to the Space), `DATABASE_URL`, `DEEPSEEK_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`. A repository variable
`SEC_EDGAR_USER_AGENT` ("Your Name you@example.com") is recommended for ingestion.

**First deployment:** run *Seed production database*, then *Deploy to Hugging Face Space*.
Re-run the seed workflow to refresh the data.

**Cost:** infrastructure is free (HF free CPU Space, Neon free tier). The only variable cost is the
LLM: about $0.001 per question on DeepSeek, and the app rate-limits clients (`RATE_LIMIT_PER_MINUTE`,
20 on the Space) to protect it. The Space sleeps after a period of inactivity and wakes on the next visit.
