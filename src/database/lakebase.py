"""Lakebase (Postgres) access for application state (spec C-2).

Connection: ``psycopg2``, using the workspace-provided credentials that ``app/app.yaml``
exposes as environment variables. No credentials in code.

Small functions, nothing clever::

    add_ticker(...)
    remove_ticker(...)
    save_report(...)
    get_watchlist(...)

Parameterized SQL only — never string-formatted SQL.

Tables owned here: ``users``, ``watchlists``, ``watchlist_tickers``, ``research_reports``.

``watchlist_tickers`` and ``research_reports`` have Lakebase CDF enabled so their changes
arrive in Delta history tables. That is the CDC demo: the agent writes a row in Postgres,
and the change is captured through the WAL into the lakehouse.
"""
