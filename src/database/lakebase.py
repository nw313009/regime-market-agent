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

HOW THE PER-CONNECTION CREDENTIAL IS INJECTED. ``ConnectionPool`` calls
``connection_class.connect(conninfo, **kwargs)`` for every new connection, and ``kwargs`` is
fixed for the life of the pool — so a password stored there would be the one static credential
this design exists to avoid. :func:`_oauth_connection_class` therefore builds a
``psycopg.Connection`` subclass whose ``connect`` mints a fresh token at that moment. Combined
with ``max_lifetime``, no connection ever outlives the credential it was opened with.

TESTABILITY. Every function takes an optional ``conn``. Passing one skips the pool entirely,
which is how the unit tests assert the SQL and its parameters without a database. When a function
takes its own connection from the pool, the pool's context manager owns the transaction and
commits on exit; when a caller injects one, the caller owns it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

__all__ = [
    "DEMO_USER_ID",
    "DEMO_WATCHLIST_ID",
    "POOL_MAX_LIFETIME_SECONDS",
    "REPORTS_TABLE",
    "SCHEMA",
    "USERS_TABLE",
    "WATCHLISTS_TABLE",
    "WATCHLIST_TICKERS_TABLE",
    "LakebaseConfigError",
    "LakebaseSettings",
    "add_ticker",
    "close_pool",
    "configure",
    "create_lakebase_sql",
    "databricks_credential_provider",
    "ensure_tables",
    "get_pool",
    "get_watchlist",
    "pool_kwargs",
    "remove_ticker",
    "save_report",
    "seed_demo",
    "settings_from_env",
]

log = logging.getLogger(__name__)

#: The schema, fixed in code rather than read from the environment. It is an IDENTIFIER, so it
#: cannot be a query parameter, and an environment-driven identifier spliced into SQL is the one
#: thing this module refuses to do.
SCHEMA = "market_system"

USERS_TABLE = f"{SCHEMA}.users"
WATCHLISTS_TABLE = f"{SCHEMA}.watchlists"
WATCHLIST_TICKERS_TABLE = f"{SCHEMA}.watchlist_tickers"
REPORTS_TABLE = f"{SCHEMA}.research_reports"

#: 50 minutes. The OAuth credential is good for at most an hour, so a connection is recycled
#: before it can be holding a dead token.
POOL_MAX_LIFETIME_SECONDS = 3000

POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4
POOL_TIMEOUT_SECONDS = 30.0
CONNECT_TIMEOUT_SECONDS = 10

#: The single demo identity (spec section 15). AMD is deliberately NOT seeded: adding it live
#: through the agent is the CDC demo moment.
DEMO_USER_ID = "demo-user"
DEMO_USER_NAME = "Demo Analyst"
DEMO_WATCHLIST_ID = "demo-watchlist"
DEMO_WATCHLIST_NAME = "My Watchlist"

#: Tickers arrive from an LLM tool call, i.e. from user text. Anything that is not a plain symbol
#: is rejected before it reaches the database.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_LAKEBASE_SQL_PATH = _REPO_ROOT / "setup" / "create_lakebase.sql"


class LakebaseConfigError(RuntimeError):
    """Connection settings are missing or unusable."""


@dataclass(frozen=True)
class LakebaseSettings:
    """Everything needed to open a connection except the credential, which is minted per use."""

    host: str
    database: str
    user: str
    endpoint: str
    """Lakebase endpoint resource name: ``projects/{project}/branches/{branch}/endpoints/{id}``."""

    port: int = 5432
    sslmode: str = "require"
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS

    def connection_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
        }


def settings_from_env(env: Mapping[str, str] | None = None) -> LakebaseSettings:
    """Read connection settings from the environment.

    Two naming schemes, because two deployments already use them: ``PG*`` for local development
    (``.env``) and ``LAKEBASE_*`` for the Databricks App (``app.yaml``). There is no password
    variable in either, and there must never be one.
    """
    source = env if env is not None else os.environ

    host = _first(source, "PGHOST", "LAKEBASE_HOST")
    database = _first(source, "PGDATABASE", "LAKEBASE_DATABASE") or "databricks_postgres"
    user = _first(source, "PGUSER", "LAKEBASE_USER")
    endpoint = _first(source, "LAKEBASE_ENDPOINT")
    port = _first(source, "PGPORT", "LAKEBASE_PORT") or "5432"
    sslmode = _first(source, "PGSSLMODE") or "require"

    missing = [
        name
        for name, value in (
            ("PGHOST / LAKEBASE_HOST", host),
            ("PGUSER / LAKEBASE_USER", user),
            ("LAKEBASE_ENDPOINT", endpoint),
        )
        if not value
    ]
    if missing:
        raise LakebaseConfigError(
            "Lakebase connection settings are incomplete: missing " + ", ".join(missing) + ". "
            "Locally these come from .env; in the deployed app they come from app.yaml. There is "
            "no password variable — the credential is minted per connection."
        )

    try:
        port_number = int(port)
    except ValueError as exc:
        raise LakebaseConfigError(f"Lakebase port {port!r} is not a number") from exc

    return LakebaseSettings(
        host=host,
        database=database,
        user=user,
        endpoint=endpoint,
        port=port_number,
        sslmode=sslmode,
    )


def databricks_credential_provider(
    endpoint: str,
    workspace_client: Any | None = None,
) -> Callable[[], str]:
    """A callable returning a fresh Postgres OAuth token for ``endpoint``.

    ``w.postgres.generate_database_credential(endpoint=...)`` is the whole authentication story:
    the caller's own workspace identity — a developer locally, the job's service principal on a
    cluster, the app's service principal once deployed — is exchanged for a short-lived Postgres
    password. The ``WorkspaceClient`` is built once and reused; the token is not.
    """
    holder: dict[str, Any] = {"client": workspace_client}

    def _token() -> str:
        client = holder["client"]
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient()
            holder["client"] = client

        credential = client.postgres.generate_database_credential(endpoint=endpoint)
        token = getattr(credential, "token", None)
        if not token:
            raise LakebaseConfigError(
                f"generate_database_credential returned no token for endpoint {endpoint!r}"
            )
        return token

    return _token


# ------------------------------------------------------------------------------ pool


_pool_lock = threading.Lock()
_pool: ConnectionPool | None = None
_settings: LakebaseSettings | None = None
_credential_provider: Callable[[], str] | None = None


def configure(
    settings: LakebaseSettings | None = None,
    *,
    credential_provider: Callable[[], str] | None = None,
) -> None:
    """Install settings and/or a credential provider, closing any pool built from the old ones."""
    global _settings, _credential_provider

    with _pool_lock:
        _settings = settings
        _credential_provider = credential_provider
    close_pool()


def pool_kwargs(settings: LakebaseSettings, credential: Callable[[], str]) -> dict:
    """Every ``ConnectionPool`` argument, as a dict.

    A pure function so the proven pattern is ASSERTABLE without a database: ``max_lifetime``
    below the credential's lifetime, ``check`` at checkout, and no password anywhere in the
    connection kwargs. Those three are the pattern; getting one of them wrong produces
    intermittent failures under load rather than an obvious break.
    """
    return {
        "conninfo": "",
        "connection_class": _oauth_connection_class(credential),
        "kwargs": settings.connection_kwargs(),
        "min_size": POOL_MIN_SIZE,
        "max_size": POOL_MAX_SIZE,
        "timeout": POOL_TIMEOUT_SECONDS,
        "max_lifetime": POOL_MAX_LIFETIME_SECONDS,
        "check": ConnectionPool.check_connection,
        "name": "lakebase-market-system",
        "open": True,
    }


def get_pool() -> ConnectionPool:
    """The process-wide connection pool, built on first use."""
    global _pool

    with _pool_lock:
        if _pool is None:
            settings = _settings or settings_from_env()
            credential = _credential_provider or databricks_credential_provider(settings.endpoint)
            _pool = ConnectionPool(**pool_kwargs(settings, credential))
            log.info(
                "lakebase pool opened host=%s database=%s user=%s max_lifetime=%d",
                settings.host,
                settings.database,
                settings.user,
                POOL_MAX_LIFETIME_SECONDS,
            )
        return _pool


def close_pool() -> None:
    """Close the pool if one is open. Safe to call when there is none."""
    global _pool

    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.close()


def _oauth_connection_class(credential: Callable[[], str]) -> type:
    """A ``psycopg.Connection`` subclass that authenticates with a freshly minted token."""

    class _OAuthConnection(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo: str = "", **kwargs: Any) -> Any:
            kwargs["password"] = credential()
            return super().connect(conninfo, **kwargs)

    return _OAuthConnection


@contextmanager
def _connection(conn: Any | None = None):
    """Yield ``conn`` when the caller supplied one, otherwise borrow one from the pool."""
    if conn is not None:
        yield conn
        return
    with get_pool().connection() as pooled:
        yield pooled


# --------------------------------------------------------------------------- schema


def create_lakebase_sql() -> str:
    """The text of ``setup/create_lakebase.sql``.

    Read rather than duplicated: a second copy of the DDL inside Python is the drift the empty
    ``sql/`` directory exists to prevent (spec B0).
    """
    try:
        return CREATE_LAKEBASE_SQL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise LakebaseConfigError(
            f"cannot read {CREATE_LAKEBASE_SQL_PATH}: {type(exc).__name__}"
        ) from exc


def ensure_tables(*, conn: Any | None = None) -> None:
    """Create the schema, the four tables and their CDF replica identity. Idempotent.

    Executes ``setup/create_lakebase.sql`` as one script — psycopg sends a statement without
    parameters over the simple query protocol, which accepts multiple statements.
    """
    with _connection(conn) as connection:
        with connection.cursor() as cur:
            cur.execute(create_lakebase_sql())
    log.info("lakebase schema ensured schema=%s", SCHEMA)


def seed_demo(
    tickers: Iterable[str] = (),
    *,
    user_id: str = DEMO_USER_ID,
    display_name: str = DEMO_USER_NAME,
    watchlist_id: str = DEMO_WATCHLIST_ID,
    watchlist_name: str = DEMO_WATCHLIST_NAME,
    conn: Any | None = None,
) -> list[str]:
    """Seed one demo user, one watchlist and optionally its tickers. Returns the watchlist.

    Idempotent through ``ON CONFLICT DO NOTHING`` on every insert, so re-running it before a demo
    cannot duplicate anything or clobber a ticker someone added by hand.
    """
    symbols = [_normalize_ticker(ticker) for ticker in tickers]

    with _connection(conn) as connection:
        with connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO {USERS_TABLE} (user_id, display_name) "
                "VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, display_name),
            )
            cur.execute(
                f"INSERT INTO {WATCHLISTS_TABLE} (watchlist_id, user_id, name) "
                "VALUES (%s, %s, %s) ON CONFLICT (watchlist_id) DO NOTHING",
                (watchlist_id, user_id, watchlist_name),
            )
            for symbol in symbols:
                cur.execute(
                    f"INSERT INTO {WATCHLIST_TICKERS_TABLE} (watchlist_id, ticker, added_by) "
                    "VALUES (%s, %s, %s) ON CONFLICT (watchlist_id, ticker) DO NOTHING",
                    (watchlist_id, symbol, user_id),
                )
            return _select_watchlist(cur, watchlist_id)


# ------------------------------------------------------------------------ watchlist


def get_watchlist(
    watchlist_id: str = DEMO_WATCHLIST_ID,
    *,
    conn: Any | None = None,
) -> list[str]:
    """Tickers on a watchlist, alphabetically."""
    with _connection(conn) as connection:
        with connection.cursor() as cur:
            return _select_watchlist(cur, watchlist_id)


def add_ticker(
    ticker: str,
    *,
    watchlist_id: str = DEMO_WATCHLIST_ID,
    added_by: str = DEMO_USER_ID,
    conn: Any | None = None,
) -> list[str]:
    """Add a ticker and return the new watchlist. Adding one twice is a no-op.

    This is the CDC demo write: the row landing here arrives in the Delta history table through
    Lakebase CDF.
    """
    symbol = _normalize_ticker(ticker)

    with _connection(conn) as connection:
        with connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO {WATCHLIST_TICKERS_TABLE} (watchlist_id, ticker, added_by) "
                "VALUES (%s, %s, %s) ON CONFLICT (watchlist_id, ticker) DO NOTHING",
                (watchlist_id, symbol, added_by),
            )
            watchlist = _select_watchlist(cur, watchlist_id)

    log.info("watchlist add watchlist_id=%s ticker=%s size=%d", watchlist_id, symbol, len(watchlist))
    return watchlist


def remove_ticker(
    ticker: str,
    *,
    watchlist_id: str = DEMO_WATCHLIST_ID,
    conn: Any | None = None,
) -> list[str]:
    """Remove a ticker and return the new watchlist. Removing an absent one is a no-op."""
    symbol = _normalize_ticker(ticker)

    with _connection(conn) as connection:
        with connection.cursor() as cur:
            cur.execute(
                f"DELETE FROM {WATCHLIST_TICKERS_TABLE} WHERE watchlist_id = %s AND ticker = %s",
                (watchlist_id, symbol),
            )
            watchlist = _select_watchlist(cur, watchlist_id)

    log.info(
        "watchlist remove watchlist_id=%s ticker=%s size=%d", watchlist_id, symbol, len(watchlist)
    )
    return watchlist


# -------------------------------------------------------------------------- reports


def save_report(
    ticker: str,
    question: str,
    report_md: str,
    *,
    user_id: str = DEMO_USER_ID,
    forecast_id: str | None = None,
    report_id: str | None = None,
    conn: Any | None = None,
) -> str:
    """Persist one agent answer and return its ``report_id``.

    ``forecast_id`` is the ``gold.forecast_runs`` row the answer was based on, so a saved report
    stays traceable to the numbers it quoted. It is a uuid5 of (ticker, as_of_date, model_used)
    and survives a re-run of the daily job (B-6), which is what makes storing it worthwhile.
    """
    symbol = _normalize_ticker(ticker)
    new_id = report_id or str(uuid.uuid4())
    if not report_md or not report_md.strip():
        raise ValueError("report_md is empty — refusing to save an empty report")

    with _connection(conn) as connection:
        with connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO {REPORTS_TABLE} "
                "(report_id, user_id, ticker, question, report_md, forecast_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (new_id, user_id, symbol, question, report_md, forecast_id),
            )

    log.info("report saved report_id=%s ticker=%s forecast_id=%s", new_id, symbol, forecast_id)
    return new_id


# ------------------------------------------------------------------------ internals


def _select_watchlist(cur: Any, watchlist_id: str) -> list[str]:
    cur.execute(
        f"SELECT ticker FROM {WATCHLIST_TICKERS_TABLE} WHERE watchlist_id = %s ORDER BY ticker",
        (watchlist_id,),
    )
    return [str(row[0]) for row in cur.fetchall() or ()]


def _normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a symbol coming from an agent tool call.

    Validation, not sanitization: the SQL is parameterized either way, so this is about keeping
    junk out of a table the demo reads aloud, and about failing on ``"add tesla to my list"``
    instead of storing it.
    """
    symbol = str(ticker or "").strip().upper()
    if not _TICKER_RE.match(symbol):
        raise ValueError(
            f"{ticker!r} is not a ticker symbol (expected 1-10 characters, A-Z start, "
            "letters/digits/./- after)"
        )
    return symbol


def _first(source: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = (source.get(name) or "").strip()
        if value:
            return value
    return ""
