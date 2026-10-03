"""SQL views that give BI tools (Power BI, Metabase, ...) tidy, ready-to-chart tables"""

BI_VIEWS = [
    # Prices rebased to 100 at each ticker's first date, plus daily returns.
    """
    CREATE OR REPLACE VIEW v_price_performance AS
    SELECT
        ticker,
        date,
        adj_close,
        volume,
        adj_close / first_value(adj_close) OVER w * 100 AS rebased,
        adj_close / lag(adj_close) OVER w - 1 AS daily_return
    FROM prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
    """,
    # One row per ticker and quarter with the headline line items side by side.
    """
    CREATE OR REPLACE VIEW v_fundamentals_key AS
    SELECT
        ticker,
        period,
        max(value) FILTER (WHERE metric = 'Total Revenue') AS total_revenue,
        max(value) FILTER (WHERE metric = 'Gross Profit') AS gross_profit,
        max(value) FILTER (WHERE metric = 'Operating Income') AS operating_income,
        max(value) FILTER (WHERE metric = 'Net Income') AS net_income,
        max(value) FILTER (WHERE metric = 'Free Cash Flow') AS free_cash_flow,
        max(value) FILTER (WHERE metric = 'Total Assets') AS total_assets
    FROM fundamentals
    GROUP BY ticker, period
    """,
    # Out-of-sample accuracy of each volatility model, per ticker and data split.
    """
    CREATE OR REPLACE VIEW v_model_accuracy AS
    SELECT
        ticker,
        split,
        model_version,
        count(*) AS n_days,
        avg(abs(actual_vol - lstm_vol)) AS lstm_mae,
        avg(abs(actual_vol - har_vol)) AS har_mae,
        avg(abs(actual_vol - naive_vol)) AS naive_mae,
        1 - avg(abs(actual_vol - lstm_vol)) / nullif(avg(abs(actual_vol - har_vol)), 0)
            AS lstm_skill_vs_har
    FROM vol_backtest
    GROUP BY ticker, split, model_version
    """,
]
