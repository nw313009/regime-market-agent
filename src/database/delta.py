"""Delta / Unity Catalog access helpers.

Two distinct access paths, deliberately:

- Pipeline code (ingestion, silver, features) reads and writes through Spark.
- The Streamlit app reads through ``databricks-sql-connector`` against a serverless SQL
  warehouse, because a Databricks App has no SparkSession.

Write rule for every pipeline table: MERGE on the declared keys, never a blind INSERT, so
re-running a task is idempotent (spec rule 6).

Declared MERGE keys:

- ``bronze.prices_raw``     -> ``(ticker, source_timestamp)``
- ``bronze.news_raw``       -> ``(article_id, ticker)``
- ``silver.daily_prices``   -> ``(ticker, trade_date)``
- ``silver.news_articles``  -> ``(article_id, ticker)``
- ``silver.daily_features`` -> ``(ticker, trade_date)``

The catalog name comes from config (``catalog: market_intel``), never hard-coded at call
sites.

These tables are tiny (roughly 2.5k rows per ticker). Do not partition them; the defaults
are correct.

WHAT THIS MODULE IMPLEMENTS is the second path only — the read side over the warehouse, used by
``src/agent/tools.py`` and the app pages. The Spark write path lives in ``src/pipelines`` and is
not duplicated here.

PARAMETERS ARE NATIVE, NOT INTERPOLATED. databricks-sql-connector 4.x binds ``:name`` markers
server-side when passed a dict, so no value is ever formatted into the SQL string. Identifiers
(catalog, schema, table) cannot be bound and are composed from config instead — which is why
``qualified()`` exists and why no caller passes a table name in from user input.

AUTHENTICATION, in the two environments that exist:

- Deployed app: no token anywhere. ``databricks.sdk.core.Config`` resolves the app service
  principal's own identity and the connector calls it per request.
- Local development: ``DATABRICKS_TOKEN`` from ``.env``, the same variable the SDK uses.

CONNECTION LIFETIME. One connection per call, closed on the way out. A cached long-lived
connection would save roughly a second per query, but a warehouse connection that has gone stale
fails at the next query rather than at checkout, and the queries here are a handful per user
turn. TODO: revisit if the app feels slow — the fix is a checked pool, like the Lakebase one, not
a bare module global.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "WarehouseConfigError",
    "WarehouseSettings",
    "configure",
    "connection",
    "http_path_for",
    "qualified",
    "query",
    "query_one",
    "settings_from_env",
]

log = logging.getLogger(__name__)


class WarehouseConfigError(RuntimeError):
    """The warehouse connection settings are missing or incomplete."""


def qualified(catalog: str, table: str) -> str:
    """``catalog.schema.table`` from a catalog and a catalog-relative name.

    Same convention as ``src/pipelines``: tables are written ``silver.news_articles`` in config
    and code, and the catalog is prefixed at the edge.
    """
    return f"{catalog}.{table}"


def http_path_for(warehouse_id: str) -> str:
    """The SQL warehouse HTTP path for a warehouse id.

    ``app.yaml`` carries ``DATABRICKS_WAREHOUSE_ID`` because that is what the Databricks App
    resource binding provides; the connector wants the path form.
    """
    return f"/sql/1.0/warehouses/{warehouse_id}"


@dataclass(frozen=True)
class WarehouseSettings:
    """Everything needed to open a warehouse connection."""

    server_hostname: str
    http_path: str
    access_token: str | None = None

    def connect_kwargs(self) -> dict:
        """Connector arguments, with the right authentication for the environment.

        A token when one is configured (local development), otherwise the workspace identity the
        app runs as. The credentials provider is a callable returning a callable, which is the
        connector's contract, and ``Config`` refreshes the underlying credential itself.
        """
        kwargs: dict[str, Any] = {
            "server_hostname": self.server_hostname,
            "http_path": self.http_path,
        }
        if self.access_token:
            kwargs["access_token"] = self.access_token
            return kwargs

        from databricks.sdk.core import Config

        cfg = Config(host=f"https://{self.server_hostname}")
        kwargs["credentials_provider"] = lambda: cfg.authenticate
        return kwargs


def _first(source: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = source.get(name)
        if value:
            return str(value).strip()
    return None


def settings_from_env(env: Mapping[str, str] | None = None) -> WarehouseSettings:
    """Read warehouse settings from the environment.

    Accepts either an explicit HTTP path or a warehouse id, because ``.env`` locally carries the
    path copied out of the warehouse's connection details while ``app.yaml`` carries the id.
    """
    source = env if env is not None else os.environ

    host = _first(source, "DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HOST")
    if host:
        host = host.removeprefix("https://").removeprefix("http://").rstrip("/")

    path = _first(source, "DATABRICKS_HTTP_PATH")
    warehouse_id = _first(source, "DATABRICKS_WAREHOUSE_ID")
    if not path and warehouse_id:
        path = http_path_for(warehouse_id)

    missing = [
        name
        for name, value in (
            ("DATABRICKS_HOST / DATABRICKS_SERVER_HOSTNAME", host),
            ("DATABRICKS_HTTP_PATH / DATABRICKS_WAREHOUSE_ID", path),
        )
        if not value
    ]
    if missing:
        raise WarehouseConfigError(
            "SQL warehouse settings are incomplete: missing " + ", ".join(missing) + ". "
            "Locally these come from .env; in the deployed app they come from app.yaml."
        )

    return WarehouseSettings(
        server_hostname=str(host),
        http_path=str(path),
        access_token=_first(source, "DATABRICKS_TOKEN"),
    )


_lock = threading.Lock()
_settings: WarehouseSettings | None = None
_connect: Any = None


def configure(
    settings: WarehouseSettings | None = None,
    *,
    connect: Any = None,
) -> None:
    """Install settings and/or a connect callable.

    ``connect`` exists for the app, which may want to hand in its own factory, and for tests,
    which hand in a fake. Passing ``None`` for both restores the defaults.
    """
    global _settings, _connect

    with _lock:
        _settings = settings
        _connect = connect


def _open(settings: WarehouseSettings | None = None) -> Any:
    with _lock:
        factory = _connect
        configured = _settings

    resolved = settings or configured or settings_from_env()
    if factory is not None:
        return factory(**resolved.connect_kwargs())

    from databricks import sql

    return sql.connect(**resolved.connect_kwargs())


@contextmanager
def connection(conn: Any = None, settings: WarehouseSettings | None = None):
    """Yield a warehouse connection, opening one only if the caller did not supply it.

    A caller-supplied connection is left open: whoever opened it owns it. That is also how the
    tests reach every query without a warehouse.
    """
    if conn is not None:
        yield conn
        return

    opened = _open(settings)
    try:
        yield opened
    finally:
        opened.close()


def _rows_as_dicts(cursor: Any) -> list[dict]:
    """Cursor rows as dicts, keyed by column name.

    Positional rows would make every caller depend on SELECT order, which is exactly the kind of
    coupling that breaks silently when a column is inserted.
    """
    description = cursor.description or ()
    columns = [column[0] for column in description]
    return [dict(zip(columns, tuple(row), strict=False)) for row in cursor.fetchall()]


def query(
    sql_text: str,
    params: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    conn: Any = None,
    settings: WarehouseSettings | None = None,
) -> list[dict]:
    """Run a parameterized SELECT and return the rows as dicts.

    ``params`` is a dict of ``:name`` bindings. The connector sends them separately from the
    statement, so nothing here interpolates a value into SQL.
    """
    with connection(conn, settings) as active:
        with active.cursor() as cursor:
            cursor.execute(sql_text, params)
            return _rows_as_dicts(cursor)


def query_one(
    sql_text: str,
    params: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    conn: Any = None,
    settings: WarehouseSettings | None = None,
) -> dict | None:
    """The first row of a query, or ``None`` when there is none.

    ``None`` rather than an exception: "this ticker has no forecast yet" is a normal state the
    agent has to be able to report truthfully, not an error.
    """
    rows = query(sql_text, params, conn=conn, settings=settings)
    return rows[0] if rows else None
