"""Agent tool tests (spec C-7).

- Write tools, as INTEGRATION tests against the real Lakebase: each of
  ``update_watchlist`` and ``save_research_report`` must produce the expected row. Mocking the
  database here would not test the thing that breaks.
- Read tools: ``get_market_forecast`` and ``search_market_news`` must return schema-valid
  payloads, including the empty case (a ticker with no forecast yet, a query with no matching
  news) — the agent has to be able to say "no relevant news" truthfully.
- Assert every tool's JSON-schema declaration matches its actual signature, since a drifted
  schema fails at model-call time rather than at import time.
- Assert all SQL is parameterized.

Manual end-to-end: run the A4 demo script once before declaring Checkpoint C frozen.

TODO: implement.
"""
