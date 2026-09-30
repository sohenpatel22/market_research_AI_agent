"""The single entrypoint the API, CLI and evals use: run the agent with tracing and scores."""

import uuid

from pydantic import BaseModel

from market_research_agent import observability
from market_research_agent.agent.graph import Dependencies, run_agent
from market_research_agent.agent.schemas import AgentAnswer
from market_research_agent.observability.langfuse import git_sha


class TracedAnswer(BaseModel):
    answer: AgentAnswer
    trace_id: str | None = None
    trace_url: str | None = None


def _model_name(llm) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown")


def ask(
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    deps: Dependencies | None = None,
    graph=None,
    tags: list[str] | None = None,
    flush_now: bool = False,
) -> TracedAnswer:
    """Answer a question. With Langfuse configured, the whole run is one trace (route, tools,
    every LLM call with tokens/cost/latency) tagged with provider, model and git sha, and the
    grader's verdict is attached as scores. Without Langfuse it is just `run_agent`."""
    if deps is None:
        from market_research_agent.agent.graph import default_dependencies

        deps = default_dependencies()

    session_id = session_id or str(uuid.uuid4())
    handler = observability.get_handler()
    provider = deps.provider or "unknown"
    attrs = {
        "session_id": session_id,
        "tags": [provider, *(tags or [])],
        "metadata": {
            "provider": provider,
            "model": _model_name(deps.llm),
            "judge_model": _model_name(deps.judge),
            "git_sha": git_sha(),
        },
        "trace_name": "agent.ask",
    }
    if user_id:
        attrs["user_id"] = user_id

    with observability.trace_context(**attrs):
        answer = run_agent(
            question,
            deps,
            thread_id=session_id,
            callbacks=[handler] if handler else [],
            graph=graph,
        )

    trace_id = getattr(handler, "last_trace_id", None) if handler else None
    trace_url = observability.post_scores(
        trace_id,
        {
            "quality_passed": answer.quality_passed,
            "grade_score": answer.grade_score,
            "retries": float(answer.retries),
            "refused": answer.refused,
            "tool_errors": float(len(answer.tool_errors)),
        },
    )
    if flush_now:
        observability.flush()
    return TracedAnswer(answer=answer, trace_id=trace_id, trace_url=trace_url)
