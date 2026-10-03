"""Gradio UI: chat with the agent, forecast explorer, market-data EDA and an About page"""

import logging

import gradio as gr
from fastapi import FastAPI

from market_research_agent import observability
from market_research_agent.agent.graph import SUPPORTED_TICKERS
from market_research_agent.agent.service import ask_stream
from market_research_agent.api.ratelimit import SlidingWindowLimiter, client_key
from market_research_agent.api.runtime import Runtime
from market_research_agent.api.streaming import iter_in_thread
from market_research_agent.config import settings
from market_research_agent.llm.factory import MissingAPIKeyError
from market_research_agent.models.forecast import ForecastError, forecast
from market_research_agent.viz import charts, formatting
from market_research_agent.viz.data import load_adj_close, load_prices

logger = logging.getLogger(__name__)

EXAMPLES = [
    "What supply chain risks does Apple report, and what is the model's AAPL volatility outlook?",
    "What does NVIDIA say about export controls on its data center products?",
    "How has MSFT stock moved over the last 90 days?",
    "What are JPMorgan's main credit risk disclosures?",
    "Does the model think XOM is likely to be up over the next month?",
]

DISCLAIMER = (
    "Research and education only, not investment advice. Answers are generated from SEC filings, "
    "market data and statistical models and may be incomplete or wrong."
)


def _limit_message(limiter: SlidingWindowLimiter, request: gr.Request | None) -> str | None:
    if request is None or request.request is None:
        return None
    retry = limiter.check(client_key(request.request))
    if retry is None:
        return None
    return f"Too many requests. Please wait about {int(retry) + 1}s and try again."


def build_demo(runtime: Runtime, limiter: SlidingWindowLimiter) -> gr.Blocks:
    def chat_fn(question: str, request: gr.Request):
        """Generator: yields (progress, answer, sources, facts, status) as the agent works"""
        question = (question or "").strip()
        empty: list = []
        if len(question) < 3:
            yield "", "Please enter a question.", empty, "", ""
            return
        if (msg := _limit_message(limiter, request)) is not None:
            yield "", msg, empty, "", ""
            return
        try:
            deps, graph = runtime.deps, runtime.graph
        except MissingAPIKeyError:
            yield "", "The language model is not configured on this server.", empty, "", ""
            return

        done: list[str] = []
        yield formatting.step_markdown(done), "", empty, "", ""
        try:
            for event in iter_in_thread(lambda: ask_stream(question, None, None, deps, graph)):
                if event["type"] == "step":
                    done.append(event["node"])
                    yield formatting.step_markdown(done), "", empty, "", ""
                    continue
                traced = event["answer"]
                answer = traced.answer
                status = formatting.status_line(answer)
                if traced.trace_url:
                    status += f" | [trace]({traced.trace_url})"
                yield (
                    formatting.step_markdown(done),
                    formatting.answer_markdown(answer),
                    formatting.source_rows(answer),
                    formatting.facts_markdown(answer),
                    status,
                )
        except Exception:  # noqa: BLE001
            logger.exception("chat failed")
            yield "", "Something went wrong while answering. Please try again.", empty, "", ""

    def forecast_fn(ticker: str, horizon_label: str, request: gr.Request):
        horizon = "1w" if horizon_label.startswith("Volatility") else "1m"
        if (msg := _limit_message(limiter, request)) is not None:
            raise gr.Error(msg)
        try:
            result = forecast(ticker, horizon)
        except ForecastError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("forecast failed")
            raise gr.Error("Forecast models are unavailable.") from exc
        prices = load_prices(ticker, days=400)
        chart = charts.forecast_chart(prices, result)
        gauge = charts.direction_gauge(result.prob_up) if result.prob_up is not None else None
        return formatting.forecast_markdown(result), chart, gauge

    def eda_fn(tickers: list[str], days: int, focus: str):
        tickers = tickers or ["AAPL"]
        wide = load_adj_close(tickers, days=int(days))
        focus = focus if focus in wide else (wide.columns[0] if len(wide.columns) else focus)
        return (
            charts.price_performance_figure(wide),
            charts.returns_distribution_figure(wide, focus),
            charts.correlation_figure(wide),
        )

    with gr.Blocks(title="Market Research Agent") as demo:
        gr.Markdown(
            "# Market Research Agent\n"
            "Ask about SEC filings, prices and model forecasts for "
            f"{', '.join(SUPPORTED_TICKERS)}. Answers are cited and quality-checked.\n\n"
            f"*{DISCLAIMER}*"
        )
        with gr.Tabs():
            with gr.Tab("Ask the agent"):
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask about a company's filings or outlook...",
                    lines=2,
                )
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear")
                gr.Examples(EXAMPLES, inputs=question, label="Try one of these")
                progress = gr.Markdown(label="Progress")
                answer = gr.Markdown(label="Answer")
                status = gr.Markdown()
                facts = gr.Markdown()
                sources = gr.Dataframe(
                    headers=formatting.SOURCE_COLUMNS,
                    label="Sources",
                    wrap=True,
                    interactive=False,
                )
                outputs = [progress, answer, sources, facts, status]
                ask_btn.click(chat_fn, question, outputs, concurrency_limit=4)
                question.submit(chat_fn, question, outputs, concurrency_limit=4)
                clear_btn.click(lambda: ("", "", "", [], "", ""), None, [question, *outputs])

            with gr.Tab("Forecast"):
                with gr.Row():
                    ticker = gr.Dropdown(SUPPORTED_TICKERS, value="AAPL", label="Ticker")
                    horizon = gr.Radio(
                        ["Volatility, next week", "Direction, next month"],
                        value="Volatility, next week",
                        label="Forecast",
                    )
                    go_btn = gr.Button("Forecast", variant="primary")
                summary = gr.Markdown()
                chart = gr.Plot(label="History and forecast")
                gauge = gr.Plot(label="Direction")
                go_btn.click(forecast_fn, [ticker, horizon], [summary, chart, gauge])

            with gr.Tab("Market data"):
                with gr.Row():
                    picks = gr.CheckboxGroup(
                        SUPPORTED_TICKERS, value=SUPPORTED_TICKERS, label="Tickers"
                    )
                    window = gr.Slider(60, 2500, value=750, step=10, label="Trading days")
                    focus = gr.Dropdown(SUPPORTED_TICKERS, value="AAPL", label="Returns for")
                eda_btn = gr.Button("Update charts")
                perf = gr.Plot(label="Relative performance")
                dist = gr.Plot(label="Return distribution")
                corr = gr.Plot(label="Correlation")
                eda_btn.click(eda_fn, [picks, window, focus], [perf, dist, corr])

            with gr.Tab("About"):
                gr.Markdown(_about_markdown())
    demo.queue(default_concurrency_limit=4, max_size=20)
    return demo


def _about_markdown() -> str:
    tracing = "on" if observability.is_enabled() else "off"
    return f"""
### How it works
A LangGraph agent routes each question, retrieves filing excerpts (hybrid vector + keyword search
with a cross-encoder reranker), runs a trained volatility/direction forecaster and whitelisted
database lookups, drafts a cited answer, then has a judge model verify it against the sources,
rewriting the search and retrying if it fails.

- **LLM provider:** `{settings.llm_provider}` (switchable: DeepSeek, OpenAI, Anthropic)
- **Tracing (Langfuse):** {tracing}
- **Forecasts:** LSTM beats HAR-RV, GARCH and ARIMA on next-week volatility (Diebold-Mariano
  p = 0.001); the next-month direction model is weak (ROC-AUC about 0.6).
- **Evaluation:** RAGAS faithfulness 0.95, answer relevancy 0.88 on a 38-question golden set.

*{DISCLAIMER}*
"""


def mount_ui(app: FastAPI, runtime: Runtime) -> FastAPI:
    """Mount the Gradio UI at `/`"""
    limiter = getattr(app.state, "limiter", None) or SlidingWindowLimiter(
        settings.rate_limit_per_minute
    )
    demo = build_demo(runtime, limiter)
    return gr.mount_gradio_app(app, demo, path="/")
