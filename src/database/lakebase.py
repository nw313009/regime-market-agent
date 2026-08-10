"""Lakebase (Postgres) access for application state (spec C-2).

This module WRAPS the proven ``db.py`` pattern copied in from a prior project. It does not
replace, rewrite or "improve" it — ``db.py`` is known to work against this Lakebase instance,
so adapt around its interface.

The pattern being wrapped:

- psycopg v3 (``psycopg[binary,pool]``), not psycopg2.
- ``psycopg_pool.ConnectionPool`` holds the connections.
- Per-connection OAuth: each connection authenticates with a short-lived credential from
  ``w.postgres.generate_database_credential`` (databricks-sdk >= 0.125.0). No static password
  exists anywhere, which is what keeps spec rule 5 true.
- ``max_lifetime=3000`` — connections are recycled before the OAuth credential expires. A pool
  that outlives its token fails intermittently under load, which is the worst way to find out.
- ``check=ConnectionPool.check_connection`` — dead connections are caught at checkout instead
  of surfacing as a query error mid-request.

Instance: the capstone has its OWN Lakebase project, ``regime-market-database`` (Autoscaling,
Postgres branch ``production``, endpoint ``primary``). It does not share the instance hosting
``ticket_system`` and ``weather_system``.

Schema: tables nonetheless live in the ``market_system`` schema, and EVERY query fully qualifies
it — ``market_system.watchlist_tickers``, never bare ``watchlist_tickers``. A dedicated project
removes the collision risk but not the reason to qualify: ``search_path`` is not a contract,
qualification keeps grants and CDF targets unambiguous, and it matches the convention used by
``ticket_system`` and ``weather_system``.

Functions exposed over the pool, nothing clever::

    add_ticker(...)
    remove_ticker(...)
    save_report(...)
    get_watchlist(...)

Parameterized SQL only — never string-formatted SQL.

Tables owned here: ``market_system.users``, ``market_system.watchlists``,
``market_system.watchlist_tickers``, ``market_system.research_reports``.

``watchlist_tickers`` and ``research_reports`` have Lakebase CDF enabled so their changes arrive
in Delta history tables. That is the CDC demo: the agent writes a row in Postgres, and the
change is captured through the WAL into the lakehouse.

Deployment gotcha (spec C-2): a new Databricks App gets a new service principal that does not
exist in Postgres yet. It needs a Postgres role created through the ``regime-market-database``
project's OAuth tab plus explicit grants on ``market_system`` before first deploy. Without them
the app fails
authentication in a way that looks like broken code and is not.
"""
