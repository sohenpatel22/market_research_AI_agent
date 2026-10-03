"""Chart builders (pure functions of DataFrames)"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from market_research_agent.models.features import vol_frame
from market_research_agent.models.forecast import ForecastResult

PRICE_COLOR = "#2563eb"
VOL_COLOR = "#0f766e"
FORECAST_COLOR = "#dc2626"
BASELINE_COLOR = "#d97706"


def forecast_chart(
    prices: pd.DataFrame, result: ForecastResult | None = None, days: int = 180
) -> go.Figure:
    """Price history (top) and realized volatility (bottom) with the model's forecast marked"""
    frame = vol_frame(prices).tail(days)
    close = prices.set_index(pd.to_datetime(prices["date"]))["adj_close"].reindex(frame.index)
    realized = np.exp(frame["log_rv5"]) * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=("Adjusted close", "Realized volatility (5-day, annualized %)"),
    )
    fig.add_trace(
        go.Scatter(x=close.index, y=close, name="Adj. close", line=dict(color=PRICE_COLOR)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=realized.index, y=realized, name="Realized vol", line=dict(color=VOL_COLOR)),
        row=2,
        col=1,
    )
    if result is not None and result.predicted_vol is not None and len(frame):
        target = frame.index[-1] + pd.Timedelta(days=7)
        for label, value, color, symbol in (
            ("LSTM forecast", result.predicted_vol, FORECAST_COLOR, "diamond"),
            ("HAR baseline", result.har_baseline_vol, BASELINE_COLOR, "circle-open"),
        ):
            if value is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[target],
                    y=[value * 100],
                    mode="markers+text",
                    text=[f"{value:.1%}"],
                    textposition="top center",
                    name=label,
                    marker=dict(color=color, size=12, symbol=symbol),
                ),
                row=2,
                col=1,
            )
    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


def direction_gauge(prob_up: float) -> go.Figure:
    """Gauge for the one-month probability of a positive return"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            number=dict(suffix="%"),
            title=dict(text="P(price higher in ~1 month)"),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=PRICE_COLOR),
                steps=[
                    dict(range=[0, 45], color="#fee2e2"),
                    dict(range=[45, 55], color="#f3f4f6"),
                    dict(range=[55, 100], color="#dcfce7"),
                ],
                threshold=dict(line=dict(color="#111827", width=3), value=50),
            ),
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def _empty_figure(message: str) -> Figure:
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    return fig


def price_performance_figure(adj_close: pd.DataFrame) -> Figure:
    """Prices rebased to 100 at the start of the window, so tickers are comparable"""
    if adj_close.empty:
        return _empty_figure("No price data. Run the ingestion first.")
    rebased = adj_close.div(adj_close.bfill().iloc[0]) * 100
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    sns.lineplot(data=rebased, dashes=False, ax=ax)
    ax.set(title="Relative performance (start = 100)", xlabel="", ylabel="Index")
    ax.legend(title="", loc="upper left", ncols=min(len(rebased.columns), 5))
    fig.tight_layout()
    return fig


def returns_distribution_figure(adj_close: pd.DataFrame, ticker: str) -> Figure:
    """Histogram + KDE of daily log returns with a normal fit, to show the fat tails"""
    if ticker not in adj_close or adj_close[ticker].dropna().size < 30:
        return _empty_figure(f"Not enough data for {ticker}.")
    returns = np.log(adj_close[ticker].dropna()).diff().dropna()
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    sns.histplot(returns, bins=60, stat="density", kde=True, color=PRICE_COLOR, ax=ax)
    grid = np.linspace(returns.min(), returns.max(), 200)
    normal = np.exp(-0.5 * ((grid - returns.mean()) / returns.std()) ** 2) / (
        returns.std() * np.sqrt(2 * np.pi)
    )
    ax.plot(grid, normal, color=FORECAST_COLOR, lw=1.5, label="Normal fit")
    ax.set(
        title=f"{ticker} daily log returns (excess kurtosis {returns.kurt():.1f})",
        xlabel="Log return",
        ylabel="Density",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def correlation_figure(adj_close: pd.DataFrame) -> Figure:
    """Heatmap of daily-return correlations between the selected tickers"""
    if adj_close.shape[1] < 2:
        return _empty_figure("Select at least two tickers to see correlations.")
    corr = np.log(adj_close).diff().dropna().corr()
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", square=True, ax=ax)
    ax.set_title("Daily return correlation")
    fig.tight_layout()
    return fig
