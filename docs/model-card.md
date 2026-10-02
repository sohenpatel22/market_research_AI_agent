# Model card: volatility and direction forecasters

For research and education only. **Not investment advice.** These are small statistical models
trained on five large-cap tickers; they are not a trading signal.

## Intended use

Provide the agent with a model-based number when a question asks about near-term outlook, and show
the model next to honest baselines. The agent always states the horizon and that the figure is an
uncertain estimate.

## Models

| | Volatility forecaster | Direction classifier |
|---|---|---|
| Target | log annualized volatility over the next 5 trading days (Parkinson high-low variance proxy) | price higher in about 21 trading days |
| Model | pooled PyTorch LSTM (window 22, hidden 32), early stopping on validation loss | logistic regression or histogram gradient boosting, chosen on validation ROC-AUC (class-balanced) |
| Features | log daily / weekly / monthly range volatility, daily return (standardized per ticker using training rows only) | momentum (1, 5, 21, 63 days), 21 and 63 day volatility, drawdown, volume z-score, RSI |
| Baselines | persistence, HAR-RV, GARCH(1,1)-t, ARIMA(1,0,1) | always-up base rate |

## Data and evaluation protocol

About 10 years of daily prices for AAPL, MSFT, NVDA, JPM and XOM. One chronological split shared by
all tickers: 70% train, 15% validation, 15% test, with the last 5 (volatility) or 21 (direction)
days of train and validation **purged** so overlapping forward-looking targets cannot leak across a
boundary. No shuffling across time. Overlapping multi-day targets make forecast errors
autocorrelated, so significance uses a Diebold-Mariano test with Newey-West (HAC) errors.

## Results (test set, most recent 15% of dates)

| Volatility model | RMSE (vol) | QLIKE (lower is better) |
|---|---|---|
| **LSTM** | 0.0840 | 0.202 |
| HAR-RV | 0.0890 | 0.223 |
| ARIMA | 0.0885 | 0.239 |
| Persistence | 0.1062 | 0.288 |
| GARCH | 0.0991 | 0.311 |

LSTM vs HAR-RV: Diebold-Mariano p = 0.001. Mean absolute error of the next-week annualized
volatility forecast: LSTM 0.052 vs HAR-RV 0.055 on the test split.

Per ticker (test, `v_model_accuracy.lstm_skill_vs_har` = 1 − LSTM MAE / HAR MAE): AAPL +7.7%,
JPM +8.6%, MSFT +6.2%, NVDA +2.2%, **XOM −5.1%**. The pooled LSTM is not uniformly better: for XOM
the simple HAR baseline wins.

| Direction classifier | ROC-AUC | F1 | accuracy | base rate |
|---|---|---|---|---|
| test | 0.62 | 0.60 | 0.55 | 0.65 |

## Limitations

- **The direction model is weak.** Accuracy is below the always-up base rate (0.65); ROC-AUC 0.62 is
  a mild tilt at best. Test labels are pooled across tickers and heavily overlapping, which
  flatters ROC-AUC somewhat. The UI says so next to every probability.
- Five tickers, one market regime; no regime-shift or crisis testing beyond what the 10 years contain.
- Parkinson volatility from daily highs and lows is a proxy for true realized volatility (no intraday data).
- Pooled training means rare tickers are not special-cased; unseen tickers are rejected, not extrapolated.
- No transaction costs, no portfolio construction, no claim of tradable alpha.

## Reproducing

`uv run python -m market_research_agent.models.train` retrains and logs to MLflow
(`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`); `python -m market_research_agent.models.publish`
writes forecasts and the backtest to Postgres.
