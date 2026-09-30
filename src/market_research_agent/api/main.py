"""FastAPI application: /chat, /chat/stream, /forecast, /tickers and /health (+ the Gradio UI).

Run locally:   uv run python -m market_research_agent.api
Interactive docs at /docs. The UI is mounted at / when `with_ui=True` (the default).
"""

import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from market_research_agent.agent.graph import SUPPORTED_TICKERS
from market_research_agent.agent.service import ask, ask_stream
from market_research_agent.api import ratelimit
from market_research_agent.api.health import app_version, check_health
from market_research_agent.api.runtime import Runtime
from market_research_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ForecastRequest,
    HealthResponse,
    TickersResponse,
)
from market_research_agent.api.streaming import sse, stream_events
from market_research_agent.config import settings
from market_research_agent.llm.factory import MissingAPIKeyError
from market_research_agent.models import registry
from market_research_agent.models.forecast import ForecastError, ForecastResult, forecast

logger = logging.getLogger("market_research_agent.api")

DESCRIPTION = (
    "Agentic research assistant over SEC filings, price data and trained forecasting models. "
    "Outputs are for research and education only, not investment advice."
)


def create_app(
    runtime: Runtime | None = None,
    limiter: ratelimit.SlidingWindowLimiter | None = None,
    with_ui: bool = True,
) -> FastAPI:
    runtime = runtime or Runtime()
    limiter = limiter or ratelimit.SlidingWindowLimiter(settings.rate_limit_per_minute)

    app = FastAPI(
        title="Market Research Agent",
        version=app_version(),
        description=DESCRIPTION,
        responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    app.state.runtime = runtime
    app.state.limiter = limiter

    @app.middleware("http")
    async def request_id_and_timing(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s in %.0f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    def rate_limited(request: Request) -> None:
        ratelimit.enforce(limiter, request)

    def llm_unavailable(exc: MissingAPIKeyError) -> HTTPException:
        return HTTPException(status_code=503, detail="The language model is not configured.")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return await run_in_threadpool(check_health)

    @app.get("/tickers", response_model=TickersResponse, tags=["system"])
    async def tickers() -> TickersResponse:
        try:
            meta = (await run_in_threadpool(registry.load_bundle))["meta"]
            return TickersResponse(tickers=sorted(meta["tickers"]), model_version=meta["version"])
        except registry.ModelNotFoundError:
            return TickersResponse(tickers=sorted(SUPPORTED_TICKERS))

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["agent"],
        dependencies=[Depends(rate_limited)],
    )
    async def chat(req: ChatRequest) -> ChatResponse:
        """Answer a question with citations, forecasts and data lookups."""
        session_id = req.session_id or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            traced = await run_in_threadpool(
                ask, req.question, session_id, None, runtime.deps, runtime.graph
            )
        except MissingAPIKeyError as exc:
            raise llm_unavailable(exc) from exc
        return ChatResponse(
            answer=traced.answer,
            trace_url=traced.trace_url,
            latency_s=round(time.perf_counter() - started, 2),
            session_id=session_id,
        )

    @app.post("/chat/stream", tags=["agent"], dependencies=[Depends(rate_limited)])
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Server-Sent Events: `step` events as the agent works, then one `final` event."""
        session_id = req.session_id or uuid.uuid4().hex
        try:
            deps, graph = runtime.deps, runtime.graph
        except MissingAPIKeyError as exc:
            raise llm_unavailable(exc) from exc

        def to_message(event: dict) -> str:
            if event["type"] == "step":
                return sse("step", {"node": event["node"], "retry": event["retry"]})
            traced = event["answer"]
            return sse(
                "final",
                ChatResponse(
                    answer=traced.answer,
                    trace_url=traced.trace_url,
                    latency_s=0.0,
                    session_id=session_id,
                ).model_dump(mode="json"),
            )

        return StreamingResponse(
            stream_events(
                lambda: ask_stream(req.question, session_id, None, deps, graph), to_message
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/forecast",
        response_model=ForecastResult,
        tags=["models"],
        dependencies=[Depends(rate_limited)],
    )
    async def forecast_endpoint(req: ForecastRequest) -> ForecastResult:
        """Run the trained models: 1w = next-week volatility, 1m = direction probability."""
        try:
            return await run_in_threadpool(forecast, req.ticker, req.horizon)
        except ForecastError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except registry.ModelNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail="Forecast models are not available."
            ) from exc

    if with_ui:
        from market_research_agent.viz.gradio_app import mount_ui

        app = mount_ui(app, runtime)
    return app
