"""Pydantic schemas for the agent: LLM structured outputs, tool I/O and the final answer."""

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

from market_research_agent.models.forecast import ForecastResult

DISCLAIMER = (
    "For research and education only; not investment advice. Figures come from SEC filings, "
    "market data and statistical models, and may be incomplete or out of date."
)


# ----------------------------------------------------------------------------- tool I/O
class RetrievedChunk(BaseModel):
    id: int
    ticker: str
    filing_type: str
    filed_date: dt.date
    section: str | None = None
    text: str
    score: float = Field(description="Reciprocal-rank-fusion (or reranker) score; higher is better")


class RetrieverInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    tickers: list[str] | None = None
    filing_types: list[str] | None = None
    k: int = Field(6, ge=1, le=20)


class DataRequest(BaseModel):
    """A whitelisted, parameterized read-only lookup against the structured tables."""

    kind: Literal["latest_price", "price_change", "fundamental"]
    ticker: str
    days: int = Field(30, ge=1, le=3650, description="Look-back window for price_change")
    metric: str | None = Field(None, description="Fundamental metric name fragment, e.g. 'revenue'")


class DataResult(BaseModel):
    kind: str
    ticker: str
    summary: str
    rows: list[dict[str, Any]] = []


# ------------------------------------------------------------------ LLM structured outputs
class RouteDecision(BaseModel):
    """What the question needs, decided before any tool is called."""

    intent: Literal["filings", "forecast", "data", "mixed", "out_of_scope"]
    tickers: list[str] = Field(default_factory=list, description="Upper-case tickers mentioned")
    use_filings: bool = Field(description="Needs text from SEC filings (10-K/10-Q)")
    search_query: str = Field("", description="Standalone keyword-rich query for filing search")
    forecasts: list[Literal["1w", "1m"]] = Field(
        default_factory=list, description="1w = next-week volatility, 1m = 1-month direction"
    )
    data_requests: list[DataRequest] = Field(default_factory=list)
    refusal_reason: str | None = Field(None, description="Set only when intent is out_of_scope")


class DraftAnswer(BaseModel):
    answer: str
    cited_source_ids: list[int] = Field(
        default_factory=list, description="Ids [S#] of filing excerpts actually used"
    )


class GradeResult(BaseModel):
    grounded: bool = Field(description="Every claim is supported by the provided context")
    relevant: bool = Field(description="The answer addresses the question asked")
    score: float = Field(ge=0, le=1)
    feedback: str = Field(description="What is missing or wrong; what to search for instead")


class RewriteResult(BaseModel):
    search_query: str


# ------------------------------------------------------------------------- final output
class Citation(BaseModel):
    source_id: int
    ticker: str
    filing_type: str
    filed_date: dt.date
    section: str | None = None
    snippet: str


class AgentAnswer(BaseModel):
    question: str
    answer: str
    sources: list[Citation] = []
    forecasts: list[ForecastResult] = []
    data: list[DataResult] = []
    quality_passed: bool
    grade_score: float | None = None
    retries: int = 0
    refused: bool = False
    tool_errors: list[str] = []
    disclaimer: str = DISCLAIMER
