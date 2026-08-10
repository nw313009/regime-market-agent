"""Databricks Workflow definition (spec C-6).

Seven tasks, in the A1 order, on a daily schedule (suggested 22:30 UTC, after US market
close), plus manual "Run now":

1. ingest_prices
2. ingest_news
3. build_silver
4. build_features
5. fit_models
6. run_forecasts
7. sync_news_index      (last)

``retries = 2`` on the ingestion tasks, since those are the ones that fail for transient
network reasons.

The walk-forward backtest is deliberately NOT in this workflow — it is a separate on-demand
job (``notebooks/10_backtest_run.py``).

The job environment needs statsmodels and exchange_calendars, which are not preinstalled.

TODO: implement.
"""
