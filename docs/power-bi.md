# Power BI dashboard recipe

A three-page dashboard on the same PostgreSQL database the agent uses: price performance,
fundamentals, and the forecasting models' output. A `.pbix` file cannot be generated from code, so
this is a precise recipe; building it takes about 30 minutes.

## 1. Prepare the data (one time)

The dashboard reads **views and published tables**, never the embedding table.

```bash
uv run python -m market_research_agent.data.ingest         # prices, fundamentals, filings (if not done)
uv run python -m market_research_agent.models.publish      # creates/refreshes model_forecasts and vol_backtest
```

`init_db` (run by both commands) creates three views: `v_price_performance` (prices rebased to 100 and
daily returns), `v_fundamentals_key` (revenue, gross/operating/net income, free cash flow, total assets
side by side per quarter) and `v_model_accuracy` (out-of-sample MAE of each volatility model by ticker and split).

## 2. Connect

Power BI Desktop > **Get data** > **PostgreSQL database**.

| Setting | Local Docker stack | Neon (production) |
|---|---|---|
| Server | `localhost:5433` | the host from Neon's *Connection details* (use the **direct**, non-pooler host), port `5432` |
| Database | `market_research_agent` | your Neon database name |
| Data connectivity mode | **Import** | **Import** |
| Credentials > Database | user `market_research_agent`, password `market_research_agent` | the Neon role and password |
| Encryption | optional | tick **Encrypt connection** (Neon requires TLS) |

Import these (tick the checkboxes in the Navigator):

`v_price_performance`, `v_fundamentals_key`, `model_forecasts`, `vol_backtest`, `v_model_accuracy`.

Do **not** import `documents` (it holds 384-dimension vectors and about 5,000 long text chunks).
Rename the queries to `Price Performance`, `Fundamentals`, `Forecasts`, `Vol Backtest`, `Model Accuracy`.

## 3. Model

1. **Home > Enter data** to make a one-column table `Tickers` with `AAPL, MSFT, NVDA, JPM, XOM`
   (or *Modeling > New table*: `Tickers = DISTINCT('Price Performance'[ticker])`).
2. **Model view**: create one-to-many relationships `Tickers[ticker]` → each of the other tables'
   `ticker` column (single direction, from `Tickers`).
3. Mark `Price Performance[date]` as a date column; in `Vol Backtest` set `date` the same way.
4. In `Forecasts`, set `predicted_vol`, `har_baseline_vol`, `current_realized_vol` and `prob_up` to
   *Percentage* format; in `Vol Backtest` / `Model Accuracy` set the volatility columns to *Percentage*.

## 4. Measures (Modeling > New measure)

```DAX
Latest Close =
CALCULATE ( MAX ( 'Price Performance'[adj_close] ),
    LASTDATE ( 'Price Performance'[date] ) )

Return 30D =
VAR LastD = MAX ( 'Price Performance'[date] )
VAR Prev =
    CALCULATE ( MAX ( 'Price Performance'[adj_close] ),
        'Price Performance'[date] <= LastD - 30,
        ALL ( 'Price Performance'[date] ) )
RETURN DIVIDE ( [Latest Close], Prev ) - 1

Annualized Vol 1M =
STDEVX.S ( DATESINPERIOD ( 'Price Performance'[date], MAX ( 'Price Performance'[date] ), -1, MONTH ),
    'Price Performance'[daily_return] ) * SQRT ( 252 )

Forecast Vol (LSTM) =
CALCULATE ( MAX ( Forecasts[predicted_vol] ), Forecasts[horizon] = "1w" )

Forecast Vol (HAR) =
CALCULATE ( MAX ( Forecasts[har_baseline_vol] ), Forecasts[horizon] = "1w" )

Prob Up 1M =
CALCULATE ( MAX ( Forecasts[prob_up] ), Forecasts[horizon] = "1m" )

LSTM MAE (test) =
CALCULATE ( AVERAGEX ( 'Vol Backtest', ABS ( 'Vol Backtest'[actual_vol] - 'Vol Backtest'[lstm_vol] ) ),
    'Vol Backtest'[split] = "test" )

HAR MAE (test) =
CALCULATE ( AVERAGEX ( 'Vol Backtest', ABS ( 'Vol Backtest'[actual_vol] - 'Vol Backtest'[har_vol] ) ),
    'Vol Backtest'[split] = "test" )

LSTM Skill vs HAR = 1 - DIVIDE ( [LSTM MAE (test)], [HAR MAE (test)] )
```

## 5. Pages

**Page 1: Prices**
- Slicer: `Tickers[ticker]` (multi-select, tile style).
- Line chart: X `Price Performance[date]`, Y `rebased`, Legend `ticker` (relative performance, start = 100).
- Cards: `Latest Close`, `Return 30D`, `Annualized Vol 1M` (filtered by the slicer).
- Column chart: `Price Performance[volume]` by month.

**Page 2: Fundamentals**
- Clustered column: X `Fundamentals[period]` (as date hierarchy: quarter), Y `total_revenue`, Legend `ticker`.
- Line chart: `net_income` by `period`, Legend `ticker`.
- Matrix: rows `ticker`, columns `period`, values `total_revenue`, `net_income`, `free_cash_flow` with a
  conditional-format data bar on `net_income`.
- Tip: format values as *Billions* (Format > Display units).

**Page 3: Model outlook**
- Table: `ticker`, `Forecast Vol (LSTM)`, `Forecast Vol (HAR)`, `Prob Up 1M`, plus `Forecasts[as_of]` and
  `model_version`; data-bar on `Prob Up 1M`.
- Gauge: value `Prob Up 1M`, min 0, max 1, target 0.5.
- Line chart (the honest-evaluation view): X `Vol Backtest[date]`, Y `actual_vol`, `lstm_vol`, `har_vol`;
  filter the visual to `split = test` and one ticker via the slicer.
- Clustered bar: `Model Accuracy[ticker]` vs `lstm_skill_vs_har` (positive means the LSTM beats HAR; XOM is negative).
- Text box: "Research and education only. Not investment advice. The direction model is weak (test ROC-AUC about 0.6)."

## 6. Refresh

Re-run `make ingest` and `python -m market_research_agent.models.publish` (the weekly GitHub workflow does
this for the production database), then **Home > Refresh** in Power BI.
