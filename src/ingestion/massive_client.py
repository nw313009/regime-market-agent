"""Massive REST API client (spec A-1).

Contract::

    class MassiveClient:
        def __init__(cfg, secret_getter): ...
        def get_daily_aggregates(ticker, start_date, end_date) -> list[dict]
        def get_news(ticker, published_after) -> list[dict]

Requirements:

- Throttle every request from ``cfg.rate_limit_per_min`` (token bucket or sleep-based).
  The limit is configurable, never hard-coded: it depends on the active Massive plan.
- Follow pagination cursors / ``next_url`` to exhaustion.
- Retry 429 and 5xx with exponential backoff plus jitter, max 5 attempts.
- Raise immediately on 401/403 with a clear message (key or plan problem, not transient).
- Log every request: url with the API key removed, status, latency.

The API key is never a literal in code. It arrives through the injected
``secret_getter`` callable, backed by Databricks secrets or the environment.

The aggregates and news route paths must be confirmed against Massive's current docs
rather than recalled from memory; the A-0 smoke test is what proves the route.
"""
