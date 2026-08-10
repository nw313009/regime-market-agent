"""Massive REST API client (spec A-1).

Contract::

    class MassiveClient:
        def __init__(cfg, secret_getter): ...
        def get_daily_aggregates(ticker, start_date, end_date) -> list[dict]
        def get_news(ticker, published_after) -> list[dict]

Requirements:

- Throttle every request from ``cfg.rate_limit_per_min`` (token bucket or sleep-based).
  The limit is configurable, never hard-coded: it depends on the active Massive plan.
- Follow pagination to exhaustion via ``next_url`` from the response envelope. Verified live:
  the envelope is ``{count, next_url, request_id, results, status}``.
- Retry 429 and 5xx with exponential backoff plus jitter, max 5 attempts.
- Raise immediately on 401/403 with a clear message (key or plan problem, not transient).
- Log every request: status, latency, endpoint name, and ``request_id`` when it parses out.

SECURITY — the API key travels as a QUERY PARAMETER, which makes both URLs and error bodies
credential-bearing. On a non-200, NEVER log or print the response body or the full URL. Log the
status code, the ``request_id`` if parseable, and the endpoint name only (e.g.
"reference/news"). Massive's error payloads and any redirect URLs can reflect request params
straight into logs, notebook output, or an agent transcript. The same rule applies in the A-0
smoke test.

The API key is never a literal in code. It arrives through the injected ``secret_getter``
callable, backed by Databricks secrets or the environment.

Route paths must be confirmed against Massive's current docs rather than recalled from memory.
Verified so far: ``/v2/reference/news`` returns 200 with the envelope above.
"""
