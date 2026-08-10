"""``bronze.prices_raw`` -> ``silver.daily_prices`` (spec A-3).

Schema: ``ticker``, ``trade_date``, ``open``, ``high``, ``low``, ``close``, ``volume``,
``vwap``.

MERGE on ``(ticker, trade_date)``.

Timestamp rule: Massive returns epoch-milliseconds. Convert to the trading date in the
exchange timezone (``America/New_York``), NOT a UTC-naive date. A UTC-naive conversion
silently shifts late-session bars onto the wrong trading day.

Bronze already parsed that epoch value into ``source_timestamp`` (the same instant), so this
build converts ``source_timestamp`` and keeps ``t_epoch_ms`` only as a deduplication tiebreak.

``volume`` is cast from the bronze DOUBLE to LONG. The vendor sends a fractional value in
scientific notation (``1.46147597081851e+08``) and share counts are whole, so the cast rounds
rather than truncating: truncation would bias every volume down by up to one share, which is
harmless in itself but is a silent, permanent edit to source data.

STRUCTURE. The transform runs in Spark (spec rule 4: Spark owns bronze -> silver -> features),
but each derivation also exists as a pure Python function here — :func:`trade_date_from_utc`,
:func:`trade_date_from_epoch_ms`, :func:`volume_to_long` — and those are what the unit tests
pin. ``tests/test_silver.py`` additionally asserts they agree with the A-2 watermark helper, so
the repo has one session-date rule rather than two that can drift.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.pipelines import latest_per_key_sql, merge_select, qualified, require_table

log = logging.getLogger(__name__)

TASK_NAME = "build_silver_prices"
SOURCE_TABLE = "bronze.prices_raw"
TARGET_TABLE = "silver.daily_prices"
MERGE_KEYS = ("ticker", "trade_date")

#: The exchange whose sessions define a trade date. One constant, shared by the SQL expression
#: below and the pure Python functions, so the timezone cannot drift between them.
EXCHANGE_TZ_NAME = "America/New_York"
EXCHANGE_TZ = ZoneInfo(EXCHANGE_TZ_NAME)

#: Session date of a bar. ``from_utc_timestamp`` reads its input relative to the Spark session
#: timezone, which :func:`main` pins to UTC before running the build.
TRADE_DATE_EXPR = f"CAST(from_utc_timestamp(source_timestamp, '{EXCHANGE_TZ_NAME}') AS DATE)"

#: Rounded, not truncated - see the module docstring.
VOLUME_EXPR = "CAST(round(volume) AS BIGINT)"

#: Ordered ``(column, expression)`` pairs. Column order matches the table DDL.
PROJECTIONS = (
    ("ticker", "ticker"),
    ("trade_date", TRADE_DATE_EXPR),
    ("open", "`open`"),
    ("high", "high"),
    ("low", "low"),
    ("close", "`close`"),
    ("volume", VOLUME_EXPR),
    ("vwap", "vwap"),
)

DAILY_PRICE_COLUMNS = tuple(name for name, _ in PROJECTIONS)

#: Latest ingestion wins when two bronze bars resolve to one session; ``t_epoch_ms`` breaks a
#: tie within a single run.
DEDUPE_ORDER_BY = "ingested_at DESC, t_epoch_ms DESC"

#: Partition expressions for the dedupe window. ``trade_date`` repeats its expression because a
#: window function cannot reference an alias from its own SELECT.
DEDUPE_PARTITION_BY = ("ticker", TRADE_DATE_EXPR)


# ------------------------------------------------------------------ pure functions


def trade_date_from_utc(instant: datetime) -> date:
    """Instant -> the exchange session date it belongs to.

    A naive datetime is treated as UTC, which is what bronze stores.
    """
    aware = instant if instant.tzinfo else instant.replace(tzinfo=timezone.utc)
    return aware.astimezone(EXCHANGE_TZ).date()


def trade_date_from_epoch_ms(epoch_ms: int | float) -> date:
    """Epoch-milliseconds -> the exchange session date.

    Daily bars arrive stamped at 00:00 America/New_York, which is 04:00Z under EDT and 05:00Z
    under EST. Converting through the timezone rather than truncating the UTC date is what keeps
    a winter bar on its own session instead of the previous one.
    """
    return trade_date_from_utc(datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc))


def volume_to_long(volume: float | int | None) -> int | None:
    """Round a fractional vendor volume to whole shares, matching SQL ``round``.

    ``math.floor(v + 0.5)`` rather than :func:`round` because Python rounds halves to even while
    SQL rounds them up; the two must agree.
    """
    if volume is None:
        return None
    return int(math.floor(float(volume) + 0.5))


def build_source_sql(catalog: str) -> str:
    """The deduplicated SELECT that feeds the MERGE."""
    return latest_per_key_sql(
        qualified(catalog, SOURCE_TABLE),
        PROJECTIONS,
        DEDUPE_PARTITION_BY,
        DEDUPE_ORDER_BY,
    )


# --------------------------------------------------------------------- entry point


def _pin_session_timezone_to_utc(spark: Any) -> None:
    """Pin ``spark.sql.session.timeZone`` to UTC for the duration of this build.

    ``from_utc_timestamp`` interprets its input relative to the session timezone. Databricks
    defaults to UTC, but a cluster or notebook that changed it would silently shift every
    trade_date by a few hours, which is exactly the class of bug the exchange-timezone rule
    exists to prevent. Pinning it makes the conversion independent of cluster configuration.
    """
    current = spark.conf.get("spark.sql.session.timeZone", "UTC")
    if current != "UTC":
        log.warning("overriding spark.sql.session.timeZone for this build was=%s now=UTC", current)
    spark.conf.set("spark.sql.session.timeZone", "UTC")


def main(spark: Any, config: Mapping) -> dict:
    """Build ``silver.daily_prices`` from ``bronze.prices_raw``.

    Callable identically from a workflow task and from a notebook cell::

        from src.pipelines import silver_prices
        silver_prices.main(spark, config)

    Returns a summary dict for notebook display. The whole of bronze is rebuilt on every run:
    the tables are tiny (~2.5k rows per ticker), and a MERGE on ``(ticker, trade_date)`` makes
    the rebuild idempotent, so no incremental bookkeeping is needed here.
    """
    catalog = str(config["catalog"])
    source_fqn = qualified(catalog, SOURCE_TABLE)
    target_fqn = qualified(catalog, TARGET_TABLE)

    require_table(spark, source_fqn)
    require_table(spark, target_fqn)
    _pin_session_timezone_to_utc(spark)

    rows_merged = merge_select(spark, target_fqn, build_source_sql(catalog), MERGE_KEYS)

    log.info("%s complete rows_merged=%d target=%s", TASK_NAME, rows_merged, target_fqn)
    return {
        "task": TASK_NAME,
        "source": source_fqn,
        "target": target_fqn,
        "rows_merged": rows_merged,
    }


def _cli() -> None:
    """Wiring only: build a session, load config, call :func:`main`.

    Kept out of ``__main__`` so the module has no import-time side effects and the workflow task
    and the notebook both call the same function.
    """
    import argparse

    import yaml
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    logging.basicConfig(level=logging.INFO)
    main(SparkSession.builder.getOrCreate(), config)


if __name__ == "__main__":
    _cli()
