"""Spark transformations: bronze -> silver -> ``silver.daily_features``.

``silver.daily_features`` is the contract between Spark and the modeling layer. Feature
scope stays deliberately small; do not expand it while the models are unvalidated.

This package module holds the write layer the silver builds share: :func:`latest_per_key_sql`
builds the deduplicated source query, :func:`merge_sql` builds the MERGE text, and
:func:`merge_select` runs it. Both SQL builders are pure functions returning strings, so the
generated SQL is unit-testable without a SparkSession.

It also owns the exchange-timezone rule every build depends on: :data:`EXCHANGE_TZ_NAME` and
:func:`session_date_expr` exist once here so no module writes its own timezone literal, and
:func:`pin_session_timezone_to_utc` makes ``from_utc_timestamp`` independent of cluster
configuration.

``qualified``, ``require_table`` and the ``bronze.ingestion_runs`` ledger helpers
(``RunRecord``, ``record_run``, ``new_run_id``, ``utc_now``, ``truncate_error``,
``STATUS_FAILED``) are re-exported from ``src.ingestion``: they are generic Unity Catalog and
audit helpers, and the frozen repo layout (spec B0) offers no module outside ``src/ingestion/``
and ``src/pipelines/`` where Spark-facing code may live, so one copy is borrowed rather than
duplicated. The ledger was introduced for the ingestion tasks at A-2 and covers the pipeline
tasks from A-4 onward, so a workflow run has one audit row per task whatever the task does.

Nothing here imports ``pyspark`` — every Spark interaction goes through the injected ``spark``
session, which keeps the pure functions importable on a machine without Spark.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from zoneinfo import ZoneInfo

from src.ingestion import (
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
    "EXCHANGE_TZ",
    "EXCHANGE_TZ_NAME",
    "STATUS_FAILED",
    "RunRecord",
    "latest_per_key_sql",
    "merge_select",
    "merge_sql",
    "new_run_id",
    "pin_session_timezone_to_utc",
    "qualified",
    "quote_identifier",
    "record_run",
    "require_table",
    "session_date_expr",
    "truncate_error",
    "utc_now",
]

log = logging.getLogger(__name__)

#: Alias for the row-number column used to deduplicate a source query.
RANK_COLUMN = "_rn"

#: The exchange whose sessions define a trade date. Declared once for the whole package: the SQL
#: expressions and the pure Python helpers must never disagree about the timezone.
EXCHANGE_TZ_NAME = "America/New_York"
EXCHANGE_TZ = ZoneInfo(EXCHANGE_TZ_NAME)


def session_date_expr(column: str) -> str:
    """SQL for "the exchange session date of this instant".

    ``from_utc_timestamp`` reads its input relative to the Spark session timezone, which
    :func:`pin_session_timezone_to_utc` fixes at UTC before any build runs.
    """
    return f"CAST(from_utc_timestamp({column}, '{EXCHANGE_TZ_NAME}') AS DATE)"


def pin_session_timezone_to_utc(spark: Any) -> None:
    """Pin ``spark.sql.session.timeZone`` to UTC for the duration of a build.

    Databricks defaults to UTC, but a cluster or notebook that changed it would silently shift
    every session date by a few hours — exactly the class of bug the exchange-timezone rule
    exists to prevent. Pinning it makes the conversion independent of cluster configuration.
    """
    current = spark.conf.get("spark.sql.session.timeZone", "UTC")
    if current != "UTC":
        log.warning("overriding spark.sql.session.timeZone for this build was=%s now=UTC", current)
    spark.conf.set("spark.sql.session.timeZone", "UTC")


def quote_identifier(name: str) -> str:
    """Backquote an identifier so column names like ``open`` cannot read as keywords."""
    return f"`{name}`"


def latest_per_key_sql(
    source_fqn: str,
    projections: Sequence[tuple[str, str]],
    partition_by: Sequence[str],
    order_by: str,
) -> str:
    """Build a SELECT that projects ``projections`` and keeps one row per key.

    ``projections`` is an ordered sequence of ``(column_name, sql_expression)``. ``partition_by``
    holds SQL EXPRESSIONS, not output aliases: a window function cannot reference an alias
    defined in its own SELECT, so a derived key such as ``trade_date`` must repeat its
    expression here.

    The deduplication is required rather than defensive. Delta fails a MERGE outright when
    several source rows match the same target row, and bronze grain is finer than silver grain
    for prices — two bronze bars could resolve to one trading session.
    """
    if not projections:
        raise ValueError("projections must not be empty")

    select_list = ",\n           ".join(
        f"{expression} AS {quote_identifier(name)}" for name, expression in projections
    )
    output_list = ", ".join(quote_identifier(name) for name, _ in projections)
    partitions = ", ".join(partition_by)

    return (
        f"SELECT {output_list}\n"
        f"FROM (\n"
        f"    SELECT {select_list},\n"
        f"           ROW_NUMBER() OVER (PARTITION BY {partitions} ORDER BY {order_by})"
        f" AS {RANK_COLUMN}\n"
        f"    FROM {source_fqn}\n"
        f")\n"
        f"WHERE {RANK_COLUMN} = 1"
    )


def merge_sql(target_fqn: str, source_sql: str, keys: Sequence[str]) -> str:
    """Build the MERGE that upserts a source query into ``target_fqn`` on ``keys``.

    Never a blind INSERT (spec rule 6): re-running a build updates matched rows in place, so the
    daily workflow's retries cannot duplicate data.
    """
    if not keys:
        raise ValueError("keys must not be empty")

    condition = " AND ".join(
        f"t.{quote_identifier(key)} = s.{quote_identifier(key)}" for key in keys
    )
    indented = "\n".join(f"    {line}" for line in source_sql.splitlines())
    return (
        f"MERGE INTO {target_fqn} AS t\n"
        f"USING (\n{indented}\n) AS s\n"
        f"ON {condition}\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


def merge_select(
    spark: Any,
    target_fqn: str,
    source_sql: str,
    keys: Sequence[str],
) -> int:
    """Run the MERGE and return the number of source rows it presented to the target.

    The count is the deduplicated source row count, not an inserted-versus-updated split:
    reporting that split would mean reading Delta's operation metrics, which the spec does not
    ask for. These tables are tiny, so the extra count query is free.
    """
    rows = spark.sql(f"SELECT count(*) AS n FROM (\n{source_sql}\n)").first()
    row_count = 0 if rows is None else int(rows["n"])

    spark.sql(merge_sql(target_fqn, source_sql, keys))

    log.info("merged rows table=%s rows=%d keys=%s", target_fqn, row_count, ",".join(keys))
    return row_count
