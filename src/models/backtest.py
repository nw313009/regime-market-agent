"""Walk-forward backtest (spec B-5). A core deliverable, not a stretch goal.

Not part of the daily job: this runs on demand as a separate job/notebook, and the app
reads the results out of ``gold.backtest_metrics``.

Origins: weekly, over the last ``cfg.backtest.n_weeks`` weeks, per ticker, each origin
requiring at least ``cfg.backtest.min_train_days`` training rows ending at T.

At each origin T:

1. Build features and ``exog_tvtp`` using data through T only.
2. Fit the ladder C -> B -> A, recording ``model_used`` and the failure reason.
3. Forecast T+5 through ``monte_carlo.run_forecast``, decaying news from ``N_T``.
4. Score against the realized 5-day return.

Parity rule: at each origin all three models fit on the identical training window.

No future price or news observation may enter fitting, regime probabilities, features,
transition variables, or initialization.

Metrics, pooled across tickers and always reported with n:

- Brier score on P(R5 > 0)
- MAE of the median return
- 80% prediction-interval coverage
- per-model fallback rate (how often C failed to converge)

Writes one row per ``(origin, ticker, model)`` to ``gold.backtest_metrics``, plus a pooled
summary table.

A backtest that looks wildly good is a leak: check the smoothed-probabilities grep test and
the TVTP lag test before believing it.
"""
