"""``sync_lakebase_history``: Lakebase (Postgres) rows captured into Delta (spec A3, C-6).

This is the CDC direction of the demo. Operational writes — a watchlist add, a saved report — land
in Postgres through the app; this task carries them into ``market_intel.gold.lb_*_history`` so the
lakehouse holds the history of what the application did. Analytical data flows the other way and
the two never loop (spec A3).

THE TRANSPORT IS LAKEHOUSE FEDERATION, and it is the second one this task has had. Both dead ends
are recorded because each is a wall a reader would otherwise walk into again:

1. NOT psycopg. The obvious implementation reuses ``src/database/lakebase.py``, and it cannot:
   psycopg 3.3.4 SIGABRTs a serverless kernel inside libpq AT IMPORT TIME, exit code 134, before
   any of our code runs (see ``requirements-databricks.txt``). A task that imports it does not
   fail, it dies. This module never imports that one and
   ``tests/test_import_boundaries.py`` enforces it.
2. NOT Spark JDBC either, which is what this task shipped with and what never once worked.
   Serverless compute CANNOT RESOLVE THE LAKEBASE ENDPOINT'S HOSTNAME — a pre-flight isolated it
   to DNS, not to credentials, not to TLS, and not to the driver. The design assumption behind the
   JDBC version was that a minted OAuth token plus ``sslmode=require`` was the whole problem;
   the problem was that there is no route from that compute to that host at all. Bundling a
   connector to work around it hits workspace policy, which is the same wall wearing a hat.
3. SO THE READ GOES THROUGH UNITY CATALOG. Lakehouse Federation exposes Postgres as a foreign
   catalog — ``regime_lakebase`` over connection ``regime_lakebase_conn`` — and the read is then
   ordinary ``spark.sql`` against a three-level name. The routing is the workspace's problem
   rather than ours, which is exactly why it works where JDBC did not.

WHAT FEDERATION COSTS, stated plainly: a UC connection stores a STATIC credential, and the OAuth
tokens this task used to mint expire hourly, so the connection cannot use one. It authenticates as
a native Postgres role ``federation_reader`` with SELECT only on ``market_system`` and a password
held in the connection object — a role that exists solely because of that constraint, and whose
blast radius is bounded to reading the two tables below. No password appears in this repository
and rule 5 still holds.

THE DELTA WRITE SIDE IS UNCHANGED by any of this, which was the point of shaping the transport as
an injected ``read`` callable: everything after it (:func:`history_rows`, the watermark read, the
MERGE) takes plain dicts and did not move when the transport was replaced.

WATERMARK, NOT FULL RELOAD. Each target's own ``MAX(watermark column)`` is the cursor — no cursor
table, because a stored cursor and the rows it describes are two things that can disagree, and the
fix for that disagreement is to recompute the cursor from the data anyway. The comparison is ``>=``
so a second row sharing the boundary timestamp cannot be skipped forever; the write is a MERGE on
the source primary key, so re-reading the boundary row updates it in place (rule 4).

A DELETE IN POSTGRES REMOVES NOTHING HERE. These are history tables: "AMD was on the watchlist" is
a fact even after AMD is removed. The app reads Lakebase when it wants the current watchlist.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

from src.ingestion import merge_rows
from src.pipelines import (
    STATUS_FAILED,
    RunRecord,
    new_run_id,
    qualified,
    record_run,
    require_table,
    truncate_error,
    utc_now,
)

__all__ = [
    "HISTORY_TABLES",
    "TASK_NAME",
    "FederatedSource",
    "HistoryTable",
    "history_rows",
    "main",
    "read_source",
    "source_from_config",
    "source_query",
    "sync_table",
    "watermark",
]

log = logging.getLogger(__name__)

TASK_NAME = "sync_lakebase_history"

#: The Postgres schema behind the foreign catalog. Federation preserves it, so the three-level name
#: is catalog.market_system.<table> and not a flattened one.
DEFAULT_SCHEMA = "market_system"


@dataclass(frozen=True)
class FederatedSource:
    """Where Postgres appears inside Unity Catalog.

    Two strings, and neither is a credential: the foreign catalog is a UC object, and everything
    about reaching the database — host, port, TLS, the ``federation_reader`` password — lives in
    the connection behind it, configured once in the workspace. That is the entire reason this
    replaced the JDBC version, which needed all of those here and could not reach the host anyway.
    """

    catalog: str
    schema: str = DEFAULT_SCHEMA

    def table_fqn(self, table: "HistoryTable") -> str:
        return f"{self.catalog}.{self.schema}.{table.source}"


@dataclass(frozen=True)
class HistoryTable:
    """One Postgres table and the Delta table that records its history."""

    source: str
    target: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    watermark_column: str
    schema_ddl: str

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Source columns plus ``captured_at``, which is this task's own contribution."""
        return (*self.columns, "captured_at")


#: The two tables spec A3 puts on the CDC path. ``users`` and ``watchlists`` are deliberately
#: absent: they are configuration the demo seeds once, not activity worth a history.
HISTORY_TABLES: tuple[HistoryTable, ...] = (
    HistoryTable(
        source="watchlist_tickers",
        target="gold.lb_watchlist_tickers_history",
        columns=("watchlist_id", "ticker", "added_at", "added_by"),
        keys=("watchlist_id", "ticker"),
        watermark_column="added_at",
        schema_ddl=(
            "watchlist_id STRING, ticker STRING, added_at TIMESTAMP, "
            "added_by STRING, captured_at TIMESTAMP"
        ),
    ),
    HistoryTable(
        source="research_reports",
        target="gold.lb_research_reports_history",
        columns=(
            "report_id",
            "user_id",
            "ticker",
            "question",
            "report_md",
            "forecast_id",
            "created_at",
        ),
        keys=("report_id",),
        watermark_column="created_at",
        schema_ddl=(
            "report_id STRING, user_id STRING, ticker STRING, question STRING, "
            "report_md STRING, forecast_id STRING, created_at TIMESTAMP, captured_at TIMESTAMP"
        ),
    ),
)


def source_from_config(config: Mapping) -> FederatedSource:
    """Read ``lakebase.federated_catalog`` and ``lakebase.schema``.

    No environment-variable override, unlike the connection facts this replaced. Those described a
    network route, which is a per-workspace thing an operator should be able to correct without a
    commit; this names a Unity Catalog object, and a job pointed at a foreign catalog that does not
    exist should say so rather than silently read a different one.
    """
    section = dict(config.get("lakebase") or {})
    catalog = str(section.get("federated_catalog") or "")

    if not catalog:
        raise ValueError(
            f"{TASK_NAME}: lakebase.federated_catalog is not set in config/config.yaml. It is the "
            "Unity Catalog foreign catalog over the Lakebase Postgres connection; the sync reads "
            "through federation because serverless compute cannot resolve the Postgres host."
        )
    return FederatedSource(
        catalog=catalog,
        schema=str(section.get("schema") or DEFAULT_SCHEMA),
    )


def source_query(table: HistoryTable, source: FederatedSource, since: datetime | None) -> str:
    """The SELECT, with the watermark in a WHERE so federation can push it down to Postgres.

    Pushed down rather than filtered in Spark because the point of a watermark is to not transfer
    the rows again — filtering after the read would move the whole table across the wire every day
    to throw most of it away. Federation forwards a predicate this simple to the database.
    """
    projection = ", ".join(table.columns)
    sql = f"SELECT {projection} FROM {source.table_fqn(table)}"
    if since is not None:
        sql += f" WHERE {table.watermark_column} >= {_timestamp_literal(since)}"
    return sql


def watermark(spark: Any, catalog: str, table: HistoryTable) -> datetime | None:
    """The newest source timestamp already captured, or ``None`` for a first run."""
    row = spark.sql(
        f"SELECT max({table.watermark_column}) AS watermark FROM {qualified(catalog, table.target)}"
    ).first()
    if row is None:
        return None
    value = row["watermark"]
    return None if value is None else _as_datetime(value)


def read_source(
    spark: Any,
    source: FederatedSource,
    table: HistoryTable,
    since: datetime | None,
) -> list[dict]:
    """Read one Postgres table through the foreign catalog and return plain dicts.

    ONE ``spark.sql`` AGAINST A THREE-LEVEL NAME is the whole transport. There is no driver here,
    no URL, no credential and no token: Unity Catalog holds all of that in the connection behind
    the foreign catalog, which is why this reaches a host that JDBC from the same compute could
    not even resolve.

    Collected to the driver deliberately: this is a watchlist and a report log, tens of rows, and
    plain dicts are what :func:`merge_rows` takes and what a test can fake without Spark.
    """
    rows = spark.sql(source_query(table, source, since)).collect()
    return [{column: row[column] for column in table.columns} for row in rows]


def history_rows(
    rows: Sequence[Mapping],
    table: HistoryTable,
    captured_at: datetime,
) -> list[dict]:
    """Stamp source rows with the capture time. PURE — the whole shaping step, in one function."""
    return [
        {**{column: row.get(column) for column in table.columns}, "captured_at": captured_at}
        for row in rows
    ]


def sync_table(
    spark: Any,
    catalog: str,
    source: FederatedSource,
    table: HistoryTable,
    *,
    captured_at: datetime,
    read: Callable[..., Sequence[Mapping]] = read_source,
) -> dict:
    """Sync one table: read the watermark, fetch what is newer, MERGE it, report what moved."""
    target_fqn = qualified(catalog, table.target)
    require_table(spark, target_fqn)

    since = watermark(spark, catalog, table)
    fetched = read(spark, source, table, since)
    rows = history_rows(fetched, table, captured_at)

    merged = merge_rows(
        spark,
        target_fqn,
        rows,
        columns=table.target_columns,
        schema_ddl=table.schema_ddl,
        keys=table.keys,
    )
    log.info(
        "%s %s -> %s since=%s rows=%d", TASK_NAME, table.source, table.target, since, merged
    )
    return {"source": table.source, "target": table.target, "since": since, "rows_merged": merged}


def main(
    spark: Any,
    config: Mapping,
    *,
    catalog: str | None = None,
    source: FederatedSource | None = None,
    read: Callable[..., Sequence[Mapping]] = read_source,
) -> dict:
    """Capture both Lakebase tables into Delta (spec C-6).

    One ``captured_at`` for the run and one ``bronze.ingestion_runs`` row for the run. A shared
    ``captured_at`` matters: it is what makes "these rows arrived in the same sync" a readable fact
    rather than a millisecond coincidence.
    """
    catalog = catalog or str(config["catalog"])
    source = source or source_from_config(config)

    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    captured_at = utc_now()
    results: list[dict] = []

    try:
        for table in HISTORY_TABLES:
            result = sync_table(
                spark,
                catalog,
                source,
                table,
                captured_at=captured_at,
                read=read,
            )
            results.append(result)
        run.rows_written = sum(int(result["rows_merged"]) for result in results)
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info("%s run_id=%s rows=%d", TASK_NAME, run.run_id, run.rows_written)
    return {
        "task": TASK_NAME,
        "run_id": run.run_id,
        "captured_at": captured_at,
        "tables": results,
        "rows_total": run.rows_written,
    }


def _timestamp_literal(value: datetime) -> str:
    """A Databricks SQL TIMESTAMP literal from a ``datetime``, and from nothing else.

    Space-separated rather than ISO-8601's ``T``, because this literal is now parsed by Databricks
    on its way to being pushed down and that is the form its documented grammar takes. The offset
    is kept: the Postgres columns are TIMESTAMPTZ and a literal without a zone would be read in
    the session's.

    The type check is the security control, the same as in ``news_recent``: this SQL is text with
    no parameter markers, so this is what keeps a value from becoming SQL.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"expected a datetime, got {type(value).__name__}: {value!r}")
    moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return f"TIMESTAMP '{moment.astimezone(timezone.utc).isoformat(sep=' ')}'"


def _as_datetime(value: Any) -> datetime:
    """Normalize what Spark hands back for a TIMESTAMP column to an aware UTC ``datetime``."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        return _as_datetime(datetime.fromisoformat(value))
    converted = getattr(value, "to_pydatetime", None)
    if converted is not None:
        return _as_datetime(converted())
    raise TypeError(f"expected a datetime, got {type(value).__name__}: {value!r}")
