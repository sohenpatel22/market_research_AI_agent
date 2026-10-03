"""Turn agent/forecast results into the Markdown and table rows the UI displays"""

from market_research_agent.agent.schemas import AgentAnswer
from market_research_agent.models.forecast import ForecastResult

STEP_LABELS = {
    "route": "Understanding the question",
    "gather": "Retrieving filings, data and forecasts",
    "generate": "Drafting the answer",
    "grade": "Checking the answer against its sources",
    "rewrite": "Answer was weak, rewriting the search and retrying",
    "finalize": "Finalizing",
    "refuse": "Declining",
}

SOURCE_COLUMNS = ["#", "Company", "Filing", "Filed", "Section", "Excerpt"]


def step_markdown(done: list[str]) -> str:
    """A checklist of the agent's progress so far"""
    if not done:
        return "Starting..."
    lines = [f"- {STEP_LABELS.get(n, n)}" for n in done]
    return "\n".join(lines)


def source_rows(answer: AgentAnswer) -> list[list]:
    return [
        [
            s.source_id,
            s.ticker,
            s.filing_type,
            str(s.filed_date),
            s.section or "",
            s.snippet.replace("\n", " ")[:220],
        ]
        for s in answer.sources
    ]


def forecast_markdown(result: ForecastResult) -> str:
    if result.horizon == "1w":
        return (
            f"**{result.ticker} next-week volatility (annualized)**  \n"
            f"LSTM forecast **{result.predicted_vol:.1%}** | HAR baseline "
            f"{result.har_baseline_vol:.1%} | recent realized {result.current_realized_vol:.1%}  \n"
            f"As of {result.as_of} (model `{result.model_version}`)"
        )
    return (
        f"**{result.ticker} probability of a higher price in ~1 month: {result.prob_up:.0%}**  \n"
        f"Weak model (test ROC-AUC about 0.6): treat as a mild tilt, not a signal. "
        f"As of {result.as_of} (model `{result.model_version}`)"
    )


def facts_markdown(answer: AgentAnswer) -> str:
    """Forecasts and database lookups the agent used, plus any tool problems"""
    parts = []
    for f in answer.forecasts:
        parts.append(forecast_markdown(f))
    for d in answer.data:
        parts.append(f"**Data:** {d.summary}")
    for e in answer.tool_errors:
        parts.append(f"*Tool problem:* {e}")
    return "\n\n".join(parts) if parts else "*No forecasts or data lookups were needed.*"


def answer_markdown(answer: AgentAnswer) -> str:
    text = answer.answer
    if answer.refused:
        return f"> {text}"
    if not answer.quality_passed:
        text += (
            "\n\n> The quality check could not fully verify this answer against the sources. "
            "Treat it with caution."
        )
    return text


def status_line(answer: AgentAnswer, latency_s: float | None = None) -> str:
    bits = []
    if answer.refused:
        bits.append("declined")
    else:
        bits.append("passed quality check" if answer.quality_passed else "quality check failed")
        if answer.grade_score is not None:
            bits.append(f"grader score {answer.grade_score:.2f}")
        if answer.retries:
            bits.append(f"{answer.retries} retr{'y' if answer.retries == 1 else 'ies'}")
    if latency_s is not None:
        bits.append(f"{latency_s:.1f}s")
    return " | ".join(bits)
