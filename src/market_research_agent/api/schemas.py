"""Request/response models for the HTTP API."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from market_research_agent.agent.schemas import AgentAnswer

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="A plain-English question")
    session_id: str | None = Field(
        None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$", description="Groups traces in Langfuse"
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must be at least 3 characters")
        return v


class ChatResponse(BaseModel):
    answer: AgentAnswer
    trace_url: str | None = None
    latency_s: float
    session_id: str


class ForecastRequest(BaseModel):
    ticker: str = Field(description="Supported ticker, e.g. AAPL")
    horizon: Literal["1w", "1m"] = Field(
        "1w", description="1w = next-week volatility, 1m = next-month direction probability"
    )

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not _TICKER_RE.match(v):
            raise ValueError("invalid ticker")
        return v


class ComponentStatus(BaseModel):
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    git_sha: str
    llm_provider: str
    components: dict[str, ComponentStatus]


class TickersResponse(BaseModel):
    tickers: list[str]
    model_version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
