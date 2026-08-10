"""Price ingestion task: Massive aggregates -> ``bronze.prices_raw`` (spec A1.1, A-2).

For every ticker in the universe (the 5 seed tickers plus all watchlist tickers), fetch
daily aggregates since the last stored bar and MERGE into ``bronze.prices_raw``.

Bronze rows keep the near-raw payload plus ``source``, ``ingested_at``, ``request_id``,
``ticker`` and ``source_timestamp``.

MERGE keys: ``(ticker, source_timestamp)``. Never a blind INSERT — every write is
idempotent so a re-run produces identical row counts.

Also records the run in ``bronze.ingestion_runs``: ``run_id``, ``task``, ``started_at``,
``finished_at``, ``status``, ``rows_written``, ``error``.

INCREMENTAL WINDOW. Empty table for a ticker -> ``massive.backfill_start_date`` from config.
Populated -> the ticker's own watermark, ``max(source_timestamp)``, mapped to its exchange
session date. The window is INCLUSIVE of that last session: re-fetching one bar per ticker per
run costs one row and repairs a bar that was ingested mid-session, and the MERGE absorbs it.

STRUCTURE. Everything above the Spark boundary is a pure function — the window arithmetic and
the row building — so the incremental logic is unit-tested without a SparkSession
(``tests/test_ingestion.py``). Spark appears only in :func:`main` and in the shared write layer
in ``src/ingestion/__init__.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.ingestion import (
    STATUS_FAILED,
    RunRecord,
    max_value,
    merge_rows,
    new_run_id,
    qualified,
    record_run,
    require_table,
    resolve_universe,
    truncate_error,
    utc_now,
)
from src.ingestion.massive_client import (
    REQUEST_ID_KEY,
    SOURCE,
    MassiveClient,
    env_secret_getter,
    redact_secrets,
)

log = logging.getLogger(__name__)

TASK_NAME = "ingest_prices"
BRONZE_TABLE = "bronze.prices_raw"
MERGE_KEYS = ("ticker", "source_timestamp")

#: The exchange whose sessions define a trade date. Never UTC-naive (spec A-3).
EXCHANGE_TZ = ZoneInfo("America/New_York")

#: Column order shared by the row builder, the staging DataFrame and the table DDL.
PRICE_COLUMNS = (
    "ticker",
    "source_timestamp",
    "t_epoch_ms",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "transactions",
    "source",
    "ingested_at",
    "request_id",
)

# `open` and `close` are backquoted so the DDL parser cannot mistake them for keywords; the
# resulting column names are still plain open/close, matching PRICE_COLUMNS and the table.
PRICE_SCHEMA_DDL = (
    "ticker STRING, source_timestamp TIMESTAMP, t_epoch_ms BIGINT, `open` DOUBLE, high DOUBLE, "
    "low DOUBLE, `close` DOUBLE, vwap DOUBLE, volume DOUBLE, transactions BIGINT, source STRING, "
    "ingested_at TIMESTAMP, request_id STRING"
)


# ------------------------------------------------------------------ pure functions


def epoch_ms_to_utc(epoch_ms: int | float) -> datetime:
    """Epoch-milliseconds -> timezone-aware UTC datetime."""
    return datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc)


def epoch_ms_to_session_date(epoch_ms: int | float) -> date:
    """Epoch-milliseconds -> the exchange session date of that bar.

    Daily bars arrive stamped at 04:00Z / 05:00Z, which is 00:00 in America/New_York, so the
    session date is the ET calendar date. Converting explicitly rather than taking the UTC date
    keeps the answer correct if the vendor ever stamps bars at a different hour.
    """
    return epoch_ms_to_utc(epoch_ms).astimezone(EXCHANGE_TZ).date()


def price_fetch_window(
    watermark: datetime | date | None,
    backfill_start_date: date,
    today: date,
) -> tuple[date, date]:
    """Resolve the inclusive ``(start_date, end_date)`` window for one ticker.

    - No watermark (empty table for this ticker) -> the configured backfill start.
    - A watermark -> that bar's session date, re-fetched so a partially ingested session is
      repaired rather than frozen.
    - Either way the window is clamped to ``today``: asking for a future ``from`` date wastes a
      request, and a backfill date in the future would otherwise invert the range.
    """
    if watermark is None:
        start = backfill_start_date
    elif isinstance(watermark, datetime):
        start = watermark.astimezone(EXCHANGE_TZ).date() if watermark.tzinfo else watermark.date()
    else:
        start = watermark

    if start > today:
        start = today
    return start, today


def build_price_rows(
    ticker: str,
    results: Iterable[Mapping],
    ingested_at: datetime,
) -> list[dict]:
    """Map near-raw aggregate bars onto ``bronze.prices_raw`` rows.

    Payload mapping (verified live): ``o/h/l/c`` -> open/high/low/close, ``v`` -> volume,
    ``vw`` -> vwap, ``n`` -> transactions, ``t`` -> ``t_epoch_ms`` and ``source_timestamp``.
    A bar without ``t`` cannot be keyed, so it is dropped with a warning rather than merged
    under a null key.
    """
    rows: list[dict] = []
    for bar in results:
        epoch_ms = bar.get("t")
        if epoch_ms is None:
            log.warning("skipping aggregate bar without t ticker=%s", ticker)
            continue
        rows.append(
            {
                "ticker": ticker,
                "source_timestamp": epoch_ms_to_utc(epoch_ms),
                "t_epoch_ms": int(epoch_ms),
                "open": _as_float(bar.get("o")),
                "high": _as_float(bar.get("h")),
                "low": _as_float(bar.get("l")),
                "close": _as_float(bar.get("c")),
                "vwap": _as_float(bar.get("vw")),
                # DOUBLE, not int: the vendor returns volume in scientific notation.
                "volume": _as_float(bar.get("v")),
                "transactions": _as_int(bar.get("n")),
                "source": SOURCE,
                "ingested_at": ingested_at,
                "request_id": bar.get(REQUEST_ID_KEY),
            }
        )
    return rows


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _backfill_start_date(config: Mapping) -> date:
    raw = (config.get("massive") or {}).get("backfill_start_date")
    if not raw:
        raise ValueError(
            "config massive.backfill_start_date is required: it is the window used when a bronze "
            "table has no rows for a ticker yet."
        )
    return date.fromisoformat(str(raw))


# --------------------------------------------------------------------- entry point


def main(
    spark: Any,
    config: Mapping,
    *,
    client: Any | None = None,
    secret_getter: Any | None = None,
    watchlist: Sequence[str] | None = None,
    today: date | None = None,
) -> dict:
    """Run the price ingestion task.

    Callable identically from a workflow task and from a notebook cell::

        from src.ingestion import ingest_prices
        ingest_prices.main(spark, config,
                           secret_getter=lambda: dbutils.secrets.get("capstone",
                                                                     "massive_api_key"))

    Returns a summary dict (``run_id``, ``rows_written``, ``tickers``) for notebook display.
    Exactly one ``bronze.ingestion_runs`` row is written per call, on success and on failure.
    """
    catalog = str(config["catalog"])
    fqn = qualified(catalog, BRONZE_TABLE)
    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    today = today or utc_now().astimezone(EXCHANGE_TZ).date()
    backfill_start = _backfill_start_date(config)
    universe = resolve_universe(config, watchlist)

    try:
        require_table(spark, fqn)
        if client is None:
            client = MassiveClient(config["massive"], secret_getter or env_secret_getter())

        for ticker in universe:
            watermark = max_value(spark, fqn, "source_timestamp", "ticker", ticker)
            start_date, end_date = price_fetch_window(watermark, backfill_start, today)
            log.info(
                "fetching aggregates ticker=%s from=%s to=%s mode=%s",
                ticker,
                start_date,
                end_date,
                "backfill" if watermark is None else "incremental",
            )
            results = client.get_daily_aggregates(ticker, start_date, end_date)
            rows = build_price_rows(ticker, results, ingested_at=utc_now())
            run.rows_written += merge_rows(
                spark,
                fqn,
                rows,
                columns=PRICE_COLUMNS,
                schema_ddl=PRICE_SCHEMA_DDL,
                keys=MERGE_KEYS,
            )
    except BaseException as exc:
        run.status = STATUS_FAILED
        # Redacted because the key is a query parameter: raw exception text can carry a
        # credential into a queryable table (A-1 security rule).
        run.error = truncate_error(redact_secrets(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s complete run_id=%s tickers=%d rows_written=%d",
        TASK_NAME,
        run.run_id,
        len(universe),
        run.rows_written,
    )
    return {"run_id": run.run_id, "rows_written": run.rows_written, "tickers": universe}


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
