"""Spark transformations: bronze -> silver -> ``silver.daily_features``, and the gold writes.

``silver.daily_features`` is the contract between Spark and the modeling layer. Feature
scope stays deliberately small; do not expand it while the models are unvalidated.

THE GOLD WRITE LAYER (B-6) LIVES HERE, at the bottom of this module. Gold rows are produced by
the pure modeling layer and persisted with Spark, and hard rule 3 puts Spark in ``src/ingestion/``
and ``src/pipelines/`` only — so ``src/models/backtest.py`` returns dataclasses and
:func:`write_gold` is what turns them into MERGEd rows. ``src/database/delta.py`` is not that
place: it documents both access paths but implements the app's non-Spark read path (C-5).
:data:`GOLD_TABLES` declares each table's columns, DDL schema and MERGE keys in one dict, so a
column added to the DDL and forgotten in a row builder fails at write time instead of arriving as
a silent NULL.

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
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.ingestion import (
    STATUS_FAILED,
    RunRecord,
    merge_rows,
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
    "GOLD_BACKTEST_METRICS",
    "GOLD_BACKTEST_SUMMARY",
    "GOLD_FORECAST_RUNS",
    "GOLD_REGIME_STATES",
    "GOLD_TABLES",
    "STATUS_FAILED",
    "GoldTable",
    "RunRecord",
    "forecast_id_for",
    "forecast_run_row",
    "latest_per_key_sql",
    "merge_select",
    "merge_sql",
    "new_run_id",
    "pin_session_timezone_to_utc",
    "qualified",
    "quote_identifier",
    "record_run",
    "regime_state_row",
    "require_table",
    "session_date_expr",
    "truncate_error",
    "utc_now",
    "write_gold",
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


# ----------------------------------------------------------------- gold write layer (B-6)


GOLD_REGIME_STATES = "gold.regime_states"
GOLD_FORECAST_RUNS = "gold.forecast_runs"
GOLD_BACKTEST_METRICS = "gold.backtest_metrics"
GOLD_BACKTEST_SUMMARY = "gold.backtest_summary"

#: Namespace for deterministic ``forecast_id`` values. Derived rather than a pasted literal so it
#: is reproducible from the string it came from.
FORECAST_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://regime-market-agent/forecast")


@dataclass(frozen=True)
class GoldTable:
    """A gold table's write contract: column order, DDL schema, and MERGE keys."""

    table: str
    columns: tuple[str, ...]
    schema_ddl: str
    keys: tuple[str, ...]


def _gold(table: str, schema: tuple[tuple[str, str], ...], keys: tuple[str, ...]) -> GoldTable:
    """Build a :class:`GoldTable` from ``(column, type)`` pairs so the two can never disagree."""
    return GoldTable(
        table=table,
        columns=tuple(name for name, _ in schema),
        schema_ddl=", ".join(f"{name} {sql_type}" for name, sql_type in schema),
        keys=keys,
    )


#: Every gold write in the project. Keyed by table name, which is what :func:`write_gold` takes.
#:
#: MERGE keys are the IDENTITY of a row, not its surrogate id. ``forecast_runs`` therefore keys on
#: ``(ticker, as_of_date, model_used)`` rather than on ``forecast_id``: a UUID key would never
#: match an existing row, which turns the MERGE the spec requires back into the blind INSERT rule 4
#: forbids, and a re-run of the daily job would duplicate the day's forecast.
GOLD_TABLES: Mapping[str, GoldTable] = {
    GOLD_REGIME_STATES: _gold(
        GOLD_REGIME_STATES,
        (
            ("ticker", "STRING"),
            ("as_of_date", "DATE"),
            ("prob_low_vol", "DOUBLE"),
            ("prob_high_vol", "DOUBLE"),
            ("low_vol_mean", "DOUBLE"),
            ("low_vol_sigma", "DOUBLE"),
            ("high_vol_mean", "DOUBLE"),
            ("high_vol_sigma", "DOUBLE"),
            ("current_news_signal", "DOUBLE"),
            ("model_used", "STRING"),
            ("model_version", "STRING"),
        ),
        ("ticker", "as_of_date"),
    ),
    GOLD_FORECAST_RUNS: _gold(
        GOLD_FORECAST_RUNS,
        (
            ("forecast_id", "STRING"),
            ("ticker", "STRING"),
            ("generated_at", "TIMESTAMP"),
            ("as_of_date", "DATE"),
            ("horizon_days", "INT"),
            ("model_used", "STRING"),
            ("current_price", "DOUBLE"),
            ("price_p10", "DOUBLE"),
            ("price_p50", "DOUBLE"),
            ("price_p90", "DOUBLE"),
            ("return_p10", "DOUBLE"),
            ("return_p50", "DOUBLE"),
            ("return_p90", "DOUBLE"),
            ("prob_positive", "DOUBLE"),
            ("prob_loss_gt_5pct", "DOUBLE"),
            ("prob_low_vol", "DOUBLE"),
            ("prob_high_vol", "DOUBLE"),
            ("n_paths", "INT"),
            ("seed", "BIGINT"),
            ("model_version", "STRING"),
        ),
        ("ticker", "as_of_date", "model_used"),
    ),
    GOLD_BACKTEST_METRICS: _gold(
        GOLD_BACKTEST_METRICS,
        (
            ("origin_date", "DATE"),
            ("ticker", "STRING"),
            ("model", "STRING"),
            ("brier", "DOUBLE"),
            ("mae", "DOUBLE"),
            ("covered_80", "BOOLEAN"),
            ("model_used", "STRING"),
            ("converged", "BOOLEAN"),
            ("failure_reason", "STRING"),
            ("realized_return", "DOUBLE"),
            ("return_p50", "DOUBLE"),
            ("prob_positive", "DOUBLE"),
        ),
        ("origin_date", "ticker", "model"),
    ),
    GOLD_BACKTEST_SUMMARY: _gold(
        GOLD_BACKTEST_SUMMARY,
        (
            ("model", "STRING"),
            ("n", "BIGINT"),
            ("n_tickers", "INT"),
            ("brier", "DOUBLE"),
            ("mae", "DOUBLE"),
            ("coverage_80", "DOUBLE"),
            ("fallback_rate", "DOUBLE"),
            ("computed_at", "TIMESTAMP"),
        ),
        ("model",),
    ),
}


def forecast_id_for(ticker: str, as_of_date: date, model_used: str) -> str:
    """A STABLE id for the forecast of one ticker, one day, one model.

    Deterministic rather than random so re-running the day is genuinely idempotent: the row is
    updated in place and anything already pointing at it — a saved research report's
    ``forecast_id`` (C-2) — still resolves. A fresh UUID per run would leave those references
    dangling on every retry.
    """
    return str(uuid.uuid5(FORECAST_NAMESPACE, f"{ticker}|{as_of_date.isoformat()}|{model_used}"))


def forecast_run_row(
    summary: Any,
    *,
    ticker: str,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> dict:
    """A ``ForecastSummary`` (spec B-4) plus the identifiers the simulation does not own.

    Duck-typed on purpose: this module reads the summary's fields and does not import the modeling
    layer, so the Spark side and the pandas side stay independently importable.
    """
    return {
        "forecast_id": forecast_id_for(ticker, as_of_date, summary.model_used),
        "ticker": ticker,
        "generated_at": generated_at or utc_now(),
        "as_of_date": as_of_date,
        "horizon_days": int(summary.horizon_days),
        "model_used": summary.model_used,
        "current_price": float(summary.current_price),
        "price_p10": float(summary.price_p10),
        "price_p50": float(summary.price_p50),
        "price_p90": float(summary.price_p90),
        "return_p10": float(summary.return_p10),
        "return_p50": float(summary.return_p50),
        "return_p90": float(summary.return_p90),
        "prob_positive": float(summary.prob_positive),
        "prob_loss_gt_5pct": float(summary.prob_loss_gt_5pct),
        "prob_low_vol": _optional_float(summary.prob_low_vol),
        "prob_high_vol": _optional_float(summary.prob_high_vol),
        "n_paths": int(summary.n_paths),
        "seed": int(summary.seed),
        "model_version": summary.model_version,
    }


def regime_state_row(
    sorted_params: Any,
    *,
    ticker: str,
    as_of_date: date,
    current_news_signal: float,
    model_used: str,
    model_version: str,
) -> dict:
    """A ``SortedParams`` (spec B-2) as one ``gold.regime_states`` row.

    The means and sigmas are stored in DECIMAL scale, matching every other number in gold —
    ``return_p10`` and friends are decimals too, so the UI applies one percentage format to the
    whole layer rather than remembering which column was estimated in percent.
    """
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "prob_low_vol": float(sorted_params.prob_low_vol),
        "prob_high_vol": float(sorted_params.prob_high_vol),
        "low_vol_mean": float(sorted_params.mus[0]),
        "low_vol_sigma": float(sorted_params.sigmas[0]),
        "high_vol_mean": float(sorted_params.mus[1]),
        "high_vol_sigma": float(sorted_params.sigmas[1]),
        "current_news_signal": float(current_news_signal),
        "model_used": model_used,
        "model_version": model_version,
    }


def write_gold(
    spark: Any,
    catalog: str,
    task: str,
    writes: Mapping[str, Sequence[Mapping]],
    *,
    run_id: str | None = None,
) -> dict:
    """MERGE gold rows and leave exactly ONE ``bronze.ingestion_runs`` row for the task.

    ``writes`` maps a table name from :data:`GOLD_TABLES` to its rows. One call per task rather
    than one per table, because the ledger's contract is one audit row per task per run (A-4): a
    task that writes ``regime_states`` and ``forecast_runs`` should leave one row saying how many
    rows it wrote, not two rows that have to be added up.

    EVERY ROW IS VALIDATED BEFORE ANY TABLE IS TOUCHED. Delta gives no transaction across tables,
    so a bad row in the second table would otherwise leave the first one updated and the run
    marked failed — the one state that is worse than failing, because it looks like success in the
    table an operator checks first.
    """
    run = RunRecord(run_id=run_id or new_run_id(), task=task, started_at=utc_now())
    written: dict[str, int] = {}

    try:
        for name, rows in writes.items():
            _validate_gold_rows(_gold_table(name), rows)

        for name, rows in writes.items():
            gold = _gold_table(name)
            fqn = qualified(catalog, gold.table)
            require_table(spark, fqn)
            written[name] = merge_rows(
                spark,
                fqn,
                list(rows),
                columns=gold.columns,
                schema_ddl=gold.schema_ddl,
                keys=gold.keys,
            )
        run.rows_written = sum(written.values())
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s wrote gold run_id=%s rows=%d tables=%s",
        task,
        run.run_id,
        run.rows_written,
        ",".join(written),
    )
    return {
        "task": task,
        "run_id": run.run_id,
        "rows_written": written,
        "rows_total": run.rows_written,
    }


def _gold_table(name: str) -> GoldTable:
    try:
        return GOLD_TABLES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown gold table {name!r}, expected one of {sorted(GOLD_TABLES)}"
        ) from exc


def _validate_gold_rows(gold: GoldTable, rows: Sequence[Mapping]) -> None:
    """Every row must carry exactly the table's columns — no missing, no extra.

    ``rows_to_tuples`` projects by column name and fills a missing key with ``None``, so without
    this a renamed field would land as a silent NULL in a published table. An extra key would
    vanish just as quietly.
    """
    expected = set(gold.columns)
    for index, row in enumerate(rows):
        present = set(row)
        if present != expected:
            raise ValueError(
                f"{gold.table} row {index} has the wrong columns: "
                f"missing {sorted(expected - present)}, unexpected {sorted(present - expected)}"
            )


def _optional_float(value: Any) -> float | None:
    """Pass ``None`` through as a NULL; a NaN is treated the same way.

    Model A reports ``None`` for its regime probabilities — it has none — and NULL is how that
    reaches Delta. NaN is folded in as well because a NaN in a published DOUBLE column is a value
    that compares false against everything, including itself, which is never what a reader wants.
    """
    if value is None:
        return None
    number = float(value)
    return None if number != number else number
