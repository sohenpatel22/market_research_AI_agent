"""LangGraph agent: route -> gather -> generate -> grade -> (rewrite -> gather ...) -> finalize.

The grade/rewrite loop is bounded by `max_retries` (and a LangGraph recursion limit), so it can
never run forever. Every LLM call returns a Pydantic object via structured output.
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from market_research_agent.agent import prompts
from market_research_agent.agent.guardrails import sanitize_excerpt
from market_research_agent.agent.schemas import (
    AgentAnswer,
    Citation,
    DataRequest,
    DataResult,
    DraftAnswer,
    GradeResult,
    RetrievedChunk,
    RewriteResult,
    RouteDecision,
)
from market_research_agent.config import settings
from market_research_agent.llm.factory import get_chat_model, get_judge_model, structured_output
from market_research_agent.models.forecast import ForecastResult

SUPPORTED_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
NODES_PER_LOOP = 4  # rewrite, gather, generate, grade
MAX_STRUCTURED_ATTEMPTS = 3
NUDGE = "Respond only by calling the provided function with the required fields."


class StructuredOutputError(RuntimeError):
    pass


class AgentState(TypedDict, total=False):
    question: str
    route: RouteDecision
    search_query: str
    retrieved_docs: list[RetrievedChunk]
    forecasts: list[ForecastResult]
    data_results: list[DataResult]
    tool_errors: list[str]
    draft_answer: DraftAnswer
    grade: GradeResult
    quality_passed: bool
    feedback: str
    retry_count: int
    final: AgentAnswer


@dataclass
class Dependencies:
    """Everything the graph needs from the outside world, injectable for tests."""

    llm: BaseChatModel
    judge: BaseChatModel
    retrieve: Callable[..., list[RetrievedChunk]]
    forecast: Callable[[str, str], ForecastResult]
    run_sql: Callable[[DataRequest], DataResult]
    provider: str | None = None
    judge_provider: str | None = None
    max_retries: int = field(default_factory=lambda: settings.agent_max_retries)
    quality_threshold: float = field(default_factory=lambda: settings.agent_quality_threshold)
    top_k: int = 6


def default_dependencies() -> Dependencies:
    from market_research_agent.agent.retriever import retrieve
    from market_research_agent.agent.tools import forecast_tool, sql_tool

    return Dependencies(
        llm=get_chat_model(),
        judge=get_judge_model(),
        retrieve=retrieve,
        forecast=forecast_tool,
        run_sql=sql_tool,
        provider=settings.llm_provider,
        judge_provider=settings.judge_provider or settings.llm_provider,
    )


# --------------------------------------------------------------------------- formatting
def _format_sources(docs: list[RetrievedChunk]) -> str:
    if not docs:
        return "(no filing excerpts retrieved)"
    # Ids are 1..n in prompt order (not database ids); finalize() maps them back.
    return "\n".join(
        f'<source id="S{i}" ticker="{d.ticker}" form="{d.filing_type}" '
        f'filed="{d.filed_date}" section="{d.section or ""}">\n'
        f"{sanitize_excerpt(d.text)}\n</source>"
        for i, d in enumerate(docs, start=1)
    )


def _format_forecasts(fs: list[ForecastResult]) -> str:
    if not fs:
        return "(none)"
    lines = []
    for f in fs:
        if f.horizon == "1w":
            lines.append(
                f"{f.ticker} next-week annualized volatility forecast: {f.predicted_vol:.1%} "
                f"(HAR baseline {f.har_baseline_vol:.1%}; "
                f"recent realized {f.current_realized_vol:.1%}; "
                f"as of {f.as_of}, model {f.model_version})"
            )
        else:
            lines.append(
                f"{f.ticker} probability of a positive return over the next month: "
                f"{f.prob_up:.0%} (weak model; as of {f.as_of})"
            )
    return "\n".join(lines)


def _format_data(rs: list[DataResult]) -> str:
    if not rs:
        return "(none)"
    return "\n".join(f"{r.summary} {r.rows}" if r.rows else r.summary for r in rs)


def _context_for_grader(state: AgentState) -> str:
    return (
        f"<filings>\n{_format_sources(state.get('retrieved_docs', []))}\n</filings>\n"
        f"<forecasts>\n{_format_forecasts(state.get('forecasts', []))}\n</forecasts>\n"
        f"<data>\n{_format_data(state.get('data_results', []))}\n</data>"
    )


# ------------------------------------------------------------------------------- graph
def build_graph(deps: Dependencies, checkpointer: Any | bool | None = None):
    """Compile the agent graph. `checkpointer=None` uses an in-memory saver, an explicit
    saver is used as given, and `False` disables checkpointing (a long-running server
    should not keep every request's state in memory)."""

    def ask(llm, provider, schema, system: str, user: str):
        """Structured LLM call. Some providers occasionally answer in plain text instead of
        calling the schema function (parsed as None), so retry with a nudge. The nudge also
        changes the prompt, so a cached bad response is not replayed."""
        runner = structured_output(llm, schema, provider)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(MAX_STRUCTURED_ATTEMPTS):
            if attempt:
                messages = [*messages[:2], HumanMessage(content=NUDGE)]
            result = runner.invoke(messages)
            if result is not None:
                return result
        raise StructuredOutputError(f"No valid {schema.__name__} after retries")

    def route(state: AgentState) -> dict:
        try:
            decision = ask(
                deps.llm,
                deps.provider,
                RouteDecision,
                prompts.get("router_system").format(tickers=", ".join(SUPPORTED_TICKERS)),
                state["question"],
            )
        except Exception:  # noqa: BLE001 - a failed router should degrade to plain filing search
            decision = RouteDecision(
                intent="filings", use_filings=True, search_query=state["question"]
            )
        decision.tickers = [t.upper() for t in decision.tickers if t.upper() in SUPPORTED_TICKERS]
        return {
            "route": decision,
            "search_query": decision.search_query or state["question"],
            "retry_count": 0,
        }

    def after_route(state: AgentState) -> str:
        return "refuse" if state["route"].intent == "out_of_scope" else "gather"

    def refuse(state: AgentState) -> dict:
        reason = state["route"].refusal_reason
        message = prompts.DEFAULT_REFUSAL if not reason else f"{reason} {prompts.DEFAULT_REFUSAL}"
        return {
            "final": AgentAnswer(
                question=state["question"], answer=message, quality_passed=True, refused=True
            )
        }

    def retrieve_filings(query: str, tickers: list[str]) -> list[RetrievedChunk]:
        """Filing search. With several companies, search each one separately and interleave the
        results so every company is represented (one company's text can otherwise fill all of
        the top-k slots and make a comparison impossible)."""
        if len(tickers) <= 1:
            return deps.retrieve(query, tickers or None, None, deps.top_k)
        per_company = max(2, math.ceil(deps.top_k / len(tickers)))
        groups = [deps.retrieve(query, [t], None, per_company) for t in tickers]
        merged: list[RetrievedChunk] = []
        seen: set[int] = set()
        for rank in range(per_company):
            for group in groups:
                if rank < len(group) and group[rank].id not in seen:
                    seen.add(group[rank].id)
                    merged.append(group[rank])
        return merged

    def gather(state: AgentState) -> dict:
        r = state["route"]
        out: dict = {}
        errors = list(state.get("tool_errors", []))
        if r.use_filings:
            try:
                out["retrieved_docs"] = retrieve_filings(state["search_query"], r.tickers)
            except Exception as e:  # noqa: BLE001
                errors.append(f"filing search failed: {e}")
                out["retrieved_docs"] = []
        # Forecasts and lookups don't depend on the search query, so run them once.
        if "forecasts" not in state:
            forecasts = []
            for ticker in r.tickers:
                for horizon in r.forecasts:
                    try:
                        forecasts.append(deps.forecast(ticker, horizon))
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"forecast {ticker}/{horizon} failed: {e}")
            out["forecasts"] = forecasts
            results = []
            for req in r.data_requests:
                try:
                    results.append(deps.run_sql(req))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"data lookup {req.kind}/{req.ticker} failed: {e}")
            out["data_results"] = results
        out["tool_errors"] = errors
        return out

    def generate(state: AgentState) -> dict:
        feedback = state.get("feedback")
        user = prompts.get("generate_user").format(
            question=state["question"],
            feedback_block=prompts.get("feedback_block").format(feedback=feedback)
            if feedback
            else "",
            sources=_format_sources(state.get("retrieved_docs", [])),
            forecasts=_format_forecasts(state.get("forecasts", [])),
            data=_format_data(state.get("data_results", [])),
        )
        try:
            draft = ask(deps.llm, deps.provider, DraftAnswer, prompts.get("generate_system"), user)
        except StructuredOutputError:
            draft = DraftAnswer(answer="I could not produce a valid answer to this question.")
        return {"draft_answer": draft}

    def grade(state: AgentState) -> dict:
        try:
            g = ask(
                deps.judge,
                deps.judge_provider,
                GradeResult,
                prompts.get("grade_system"),
                prompts.get("grade_user").format(
                    question=state["question"],
                    context=_context_for_grader(state),
                    answer=state["draft_answer"].answer,
                ),
            )
        except Exception as e:  # noqa: BLE001 - retrying can't fix a broken grader; stop here
            return {
                "quality_passed": False,
                "feedback": f"grading unavailable: {e}",
                "retry_count": deps.max_retries,
            }
        passed = g.grounded and g.relevant and g.score >= deps.quality_threshold
        return {"grade": g, "quality_passed": passed, "feedback": g.feedback}

    def after_grade(state: AgentState) -> str:
        if state["quality_passed"]:
            return "finalize"
        can_retry = state["retry_count"] < deps.max_retries and state["route"].use_filings
        return "rewrite" if can_retry else "finalize"

    def rewrite(state: AgentState) -> dict:
        new = ask(
            deps.llm,
            deps.provider,
            RewriteResult,
            prompts.get("rewrite_system"),
            prompts.get("rewrite_user").format(
                question=state["question"],
                previous_query=state["search_query"],
                feedback=state.get("feedback", ""),
            ),
        )
        return {
            "search_query": new.search_query or state["search_query"],
            "retry_count": state["retry_count"] + 1,
        }

    def finalize(state: AgentState) -> dict:
        grade = state.get("grade")
        docs = dict(enumerate(state.get("retrieved_docs", []), start=1))
        draft = state["draft_answer"]
        # Citation verification: keep only ids that were really retrieved, once each.
        sources, seen = [], set()
        for sid in draft.cited_source_ids:
            doc = docs.get(sid)
            if doc is None or sid in seen:
                continue
            seen.add(sid)
            sources.append(
                Citation(
                    source_id=sid,
                    ticker=doc.ticker,
                    filing_type=doc.filing_type,
                    filed_date=doc.filed_date,
                    section=doc.section,
                    snippet=sanitize_excerpt(doc.text)[:300],
                )
            )
        return {
            "final": AgentAnswer(
                question=state["question"],
                answer=draft.answer,
                sources=sources,
                retrieved_context=state.get("retrieved_docs", []),
                forecasts=state.get("forecasts", []),
                data=state.get("data_results", []),
                quality_passed=state["quality_passed"],
                grade_score=grade.score if grade else None,
                retries=state["retry_count"],
                tool_errors=state.get("tool_errors", []),
            )
        }

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("refuse", refuse),
        ("gather", gather),
        ("generate", generate),
        ("grade", grade),
        ("rewrite", rewrite),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"refuse": "refuse", "gather": "gather"})
    g.add_edge("gather", "generate")
    g.add_edge("generate", "grade")
    g.add_conditional_edges("grade", after_grade, {"rewrite": "rewrite", "finalize": "finalize"})
    g.add_edge("rewrite", "gather")
    g.add_edge("finalize", END)
    g.add_edge("refuse", END)
    saver = MemorySaver() if checkpointer is None else (checkpointer or None)
    return g.compile(checkpointer=saver)


def recursion_limit(max_retries: int) -> int:
    """Upper bound on graph steps: fixed nodes + each retry loop, with headroom."""
    return 6 + NODES_PER_LOOP * max_retries + 4


def _run_config(
    deps: Dependencies, thread_id: str, callbacks: list | None, metadata: dict | None
) -> dict:
    return {
        "recursion_limit": recursion_limit(deps.max_retries),
        "callbacks": callbacks or [],
        "metadata": metadata or {},
        "configurable": {"thread_id": thread_id},
    }


def run_agent(
    question: str,
    deps: Dependencies | None = None,
    thread_id: str = "default",
    callbacks: list | None = None,
    metadata: dict | None = None,
    graph=None,
) -> AgentAnswer:
    """Answer one question. `callbacks`/`metadata` carry observability (e.g. Langfuse)."""
    deps = deps or default_dependencies()
    graph = graph or build_graph(deps)
    config = _run_config(deps, thread_id, callbacks, metadata)
    state = graph.invoke({"question": question}, config=config)
    return state["final"]


def stream_agent(
    question: str,
    deps: Dependencies | None = None,
    thread_id: str = "default",
    callbacks: list | None = None,
    metadata: dict | None = None,
    graph=None,
) -> Iterator[dict]:
    """Like run_agent, but yields progress as the graph executes.

    Events: {"type": "step", "node": <name>, "retry": <n>} after each node finishes, then a
    single {"type": "final", "answer": AgentAnswer}. (Answers come from structured output, so
    progress is streamed per graph step rather than per token.)
    """
    deps = deps or default_dependencies()
    graph = graph or build_graph(deps)
    config = _run_config(deps, thread_id, callbacks, metadata)
    retries = 0
    for update in graph.stream({"question": question}, config=config, stream_mode="updates"):
        for node, delta in update.items():
            if not isinstance(delta, dict):
                continue
            retries = delta.get("retry_count", retries)
            if "final" in delta:
                yield {"type": "final", "answer": delta["final"]}
            else:
                yield {"type": "step", "node": node, "retry": retries}
