"""Research Agent page (spec A2, C-5).

A chat box scoped to a ticker, driving the agent loop in ``src/agent/agent.py``. The
watchlist sidebar is read from Lakebase.

Supported user actions beyond plain questions:

- "Save this as a report" -> ``save_research_report``
- "Add NVDA to my watchlist" -> ``update_watchlist``

The agent explains the Gold numbers and the retrieved articles. It never computes
statistics, and it confirms any write it performs.
"""
