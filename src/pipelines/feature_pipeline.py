"""``silver.daily_prices`` + ``silver.news_articles`` -> ``silver.daily_features`` (spec A-4).

Grain: ``(ticker, trade_date)``, trading days only, from the XNYS calendar via
``exchange_calendars``. This table is the contract between Spark and the modeling layer, which
reads it with a single ``.toPandas()`` at the ticker level (spec C-b).

Columns and definitions, exactly as specified::

    log_return        = ln(close / lag(close))
    return_5d         = close / lag(close, 5) - 1
    momentum_5d       = sum of the last 5 log_returns
    realized_vol_20d  = stddev_samp(log_return) over the trailing 20 rows
    volume_zscore_20d = (volume - mean_20) / stddev_20
    s_t               = mean sentiment_score of the articles mapped to that session, 0 if none
    news_sentiment_3d = (1.0*s_t + 0.5*s_{t-1} + 0.25*s_{t-2}) / 1.75
    news_count        = number of articles mapped to that session

All rolling columns are Spark window functions partitioned by ticker, ordered by trade_date.
Warm-up rows keep NULL rolling features and stay in the table; the modeling layer drops them
(spec B-0).

NULL IS EXPLICIT, NOT INCIDENTAL. Spark's aggregates skip NULLs, so a 20-row window containing
19 usable returns would otherwise report a 19-observation standard deviation as though it were a
20-day one. Every rolling column is therefore guarded by a count over the same frame and is NULL
until the frame is genuinely full. That makes the warm-up boundary exact and testable: momentum
starts at row 5, realized vol at row 20, and the volume z-score at row 19 — one row earlier,
because volume has no undefined first row the way log_return does.

NEWS SESSION ASSIGNMENT USES THE CALENDAR, NEVER WEEKDAY ARITHMETIC. Each article's
``published_at`` is resolved to its America/New_York date and then to the first XNYS session on
or after that date. Weekday arithmetic gets weekends right and holidays wrong: Good Friday
2026-04-03 is a Friday, and an article published that day belongs to Monday 2026-04-06.
:func:`next_session_map` builds the date -> session lookup on the driver and it is broadcast to
Spark as a small table, so the join is a plain equality join and the calendar is consulted in
exactly one place.

Missing lags in ``news_sentiment_3d`` are treated as 0 rather than NULL: the spec fixes the
denominator at 1.75, and "no articles" is a real zero (A-3 already defines ``s_t`` as 0 when a
session has no news), unlike a rolling statistic that is genuinely undefined during warm-up.

STRUCTURE. The transform runs in Spark (spec rule 4), and every formula also exists as a pure
Python function here, generated from the same constants the SQL is generated from -
:data:`NEWS_DECAY_WEIGHTS`, :data:`MOMENTUM_WINDOW`, :data:`VOL_WINDOW`, :data:`VOLUME_WINDOW`,
:data:`RETURN_LAG`. ``tests/test_features.py`` pins the Python side against hand-computed values;
a row-by-row comparison of the SQL against it is on the workspace integration list.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.pipelines import (
    STATUS_FAILED,
    EXCHANGE_TZ,
    RunRecord,
    merge_select,
    new_run_id,
    pin_session_timezone_to_utc,
    qualified,
    quote_identifier,
    record_run,
    require_table,
    session_date_expr,
    truncate_error,
    utc_now,
)

log = logging.getLogger(__name__)

TASK_NAME = "build_features"
PRICES_TABLE = "silver.daily_prices"
NEWS_TABLE = "silver.news_articles"
TARGET_TABLE = "silver.daily_features"
MERGE_KEYS = ("ticker", "trade_date")

#: Trading calendar. XNYS, per spec A-4 — not a weekday rule, and not a per-ticker calendar.
CALENDAR_NAME = "XNYS"

#: Formula constants. The SQL expressions below are generated from these, so the Python
#: reference implementations and the executed SQL cannot disagree about a window length or a
#: decay weight.
RETURN_LAG = 5
MOMENTUM_WINDOW = 5
VOL_WINDOW = 20
VOLUME_WINDOW = 20

#: s_t, s_{t-1}, s_{t-2} weights and their fixed denominator (1.75), per spec A-4.
NEWS_DECAY_WEIGHTS = (1.0, 0.5, 0.25)
NEWS_DECAY_DENOMINATOR = sum(NEWS_DECAY_WEIGHTS)

#: Temp view names for the two driver-built calendar tables.
SESSIONS_VIEW = "_xnys_sessions"
SESSION_MAP_VIEW = "_xnys_session_map"
SESSIONS_SCHEMA_DDL = "session_date DATE"
SESSION_MAP_SCHEMA_DDL = "calendar_date DATE, session_date DATE"


# ------------------------------------------------------- pure functions: the calendar


def trading_sessions(start: date, end: date, calendar_name: str = CALENDAR_NAME) -> list[date]:
    """XNYS sessions in ``[start, end]``, ascending.

    ``exchange_calendars`` is imported lazily so importing this module stays cheap and does not
    require the calendar package to be installed to read the pure formula helpers.
    """
    import exchange_calendars as xcals

    calendar = xcals.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(start.isoformat(), end.isoformat())
    return [session.date() for session in sessions]


def next_session_map(
    sessions: Sequence[date],
    start: date,
    end: date,
) -> dict[date, date]:
    """Map every calendar date in ``[start, end]`` to the first session on or after it.

    This is the whole of the "roll forward to the NEXT session" rule (spec A-4), and it is a
    pure function over a session list so it can be tested against both a synthetic calendar and
    the real XNYS one. Dates after the last known session are omitted rather than guessed: an
    article published after the calendar window has no session to belong to yet.
    """
    ordered = sorted(sessions)
    mapping: dict[date, date] = {}
    index = 0
    current = start

    while current <= end:
        while index < len(ordered) and ordered[index] < current:
            index += 1
        if index >= len(ordered):
            break
        mapping[current] = ordered[index]
        current += timedelta(days=1)
    return mapping


def to_exchange_date(instant: datetime) -> date:
    """Instant -> its America/New_York calendar date. Naive input is treated as UTC."""
    aware = instant if instant.tzinfo else instant.replace(tzinfo=timezone.utc)
    return aware.astimezone(EXCHANGE_TZ).date()


def session_for_timestamp(instant: datetime, session_map: Mapping[date, date]) -> date | None:
    """The trading session an article belongs to, or ``None`` if it is past the calendar."""
    return session_map.get(to_exchange_date(instant))


def session_news_aggregates(
    articles: Iterable[Mapping],
    session_map: Mapping[date, date],
) -> dict[tuple[str, date], tuple[float, int]]:
    """Aggregate articles into ``{(ticker, session): (s_t, news_count)}``.

    ``s_t`` is the mean ``sentiment_score`` of the articles assigned to that session, which is
    how a Saturday article and a Sunday article both land in Monday's value. Rows whose
    timestamp falls past the calendar window are skipped with a warning rather than dropped
    silently.
    """
    grouped: dict[tuple[str, date], list[float]] = {}
    for article in articles:
        session = session_for_timestamp(article["published_at"], session_map)
        if session is None:
            log.warning(
                "news article has no session in the calendar window, skipping ticker=%s at=%s",
                article.get("ticker"),
                article.get("published_at"),
            )
            continue
        key = (str(article["ticker"]), session)
        grouped.setdefault(key, []).append(float(article.get("sentiment_score") or 0))

    return {key: (sum(scores) / len(scores), len(scores)) for key, scores in grouped.items()}


# ------------------------------------------- pure functions: the rolling formulas


def log_returns(closes: Sequence[float | None]) -> list[float | None]:
    """``ln(close / lag(close))``. NULL on the first row, which has no previous close."""
    out: list[float | None] = [None]
    for previous, current in zip(closes, closes[1:]):
        usable = _positive(previous) and _positive(current)
        out.append(math.log(current / previous) if usable else None)
    return out


def returns_over_lag(
    closes: Sequence[float | None],
    lag: int = RETURN_LAG,
) -> list[float | None]:
    """``close / lag(close, lag) - 1``. NULL until ``lag`` rows of history exist."""
    out: list[float | None] = []
    for index, current in enumerate(closes):
        previous = closes[index - lag] if index >= lag else None
        usable = _positive(previous) and _positive(current)
        out.append(current / previous - 1 if usable else None)
    return out


def rolling_sum(
    values: Sequence[float | None],
    window: int = MOMENTUM_WINDOW,
) -> list[float | None]:
    """Sum over the trailing ``window`` rows, NULL unless every row in the frame is present.

    The completeness guard is the point: summing 4 of 5 returns and calling it 5-day momentum is
    the kind of quiet error a warm-up period hides.
    """
    return [
        None if frame is None else math.fsum(frame)
        for frame in (_full_frame(values, index, window) for index in range(len(values)))
    ]


def rolling_stddev_samp(
    values: Sequence[float | None],
    window: int = VOL_WINDOW,
) -> list[float | None]:
    """Sample standard deviation (n-1) over the trailing ``window`` rows, NULL until full."""
    out: list[float | None] = []
    for index in range(len(values)):
        frame = _full_frame(values, index, window)
        out.append(None if frame is None else _stddev_samp(frame))
    return out


def rolling_zscore(
    values: Sequence[float | None],
    window: int = VOLUME_WINDOW,
) -> list[float | None]:
    """``(value - mean) / stddev_samp`` over the trailing ``window`` rows, inclusive of the row.

    Including the current row is the same frame ``realized_vol_20d`` uses, and it is the usual
    reading of a rolling z-score. A constant window has zero deviation, so the result is NULL
    rather than an infinity — matching ``try_divide`` in the generated SQL.
    """
    out: list[float | None] = []
    for index in range(len(values)):
        frame = _full_frame(values, index, window)
        if frame is None:
            out.append(None)
            continue
        deviation = _stddev_samp(frame)
        mean = math.fsum(frame) / len(frame)
        out.append(None if deviation == 0 else (frame[-1] - mean) / deviation)
    return out


def news_sentiment_3d(
    s_values: Sequence[float],
    weights: Sequence[float] = NEWS_DECAY_WEIGHTS,
) -> list[float]:
    """``(1.0*s_t + 0.5*s_{t-1} + 0.25*s_{t-2}) / 1.75``, missing lags counted as 0."""
    denominator = sum(weights)
    out: list[float] = []
    for index in range(len(s_values)):
        weighted = math.fsum(
            weight * (float(s_values[index - offset]) if index - offset >= 0 else 0.0)
            for offset, weight in enumerate(weights)
        )
        out.append(weighted / denominator)
    return out


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def _full_frame(
    values: Sequence[float | None],
    index: int,
    window: int,
) -> list[float] | None:
    """The trailing ``window`` values ending at ``index``, or ``None`` if incomplete."""
    if index + 1 < window:
        return None
    frame = values[index + 1 - window : index + 1]
    if any(value is None for value in frame):
        return None
    return [float(value) for value in frame]


def _stddev_samp(frame: Sequence[float]) -> float:
    mean = math.fsum(frame) / len(frame)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in frame) / (len(frame) - 1))


# --------------------------------------------------------- generated SQL expressions


def _trailing_frame(window: int) -> str:
    return f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"


def log_return_expr() -> str:
    """``try_divide`` rather than ``/`` so a zero or missing previous close yields NULL in both
    ANSI and legacy modes instead of raising."""
    return "ln(try_divide(`close`, lag(`close`, 1) OVER w_order))"


def return_over_lag_expr() -> str:
    return f"try_divide(`close`, lag(`close`, {RETURN_LAG}) OVER w_order) - 1"


def momentum_expr() -> str:
    return (
        f"CASE WHEN count(log_return) OVER w_momentum = {MOMENTUM_WINDOW}"
        " THEN sum(log_return) OVER w_momentum END"
    )


def realized_vol_expr() -> str:
    return (
        f"CASE WHEN count(log_return) OVER w_vol = {VOL_WINDOW}"
        " THEN stddev_samp(log_return) OVER w_vol END"
    )


def volume_zscore_expr() -> str:
    return (
        f"CASE WHEN count(volume) OVER w_volume = {VOLUME_WINDOW}"
        " THEN try_divide(volume - avg(volume) OVER w_volume,"
        " stddev_samp(volume) OVER w_volume) END"
    )


def news_sentiment_3d_expr() -> str:
    """Generated from :data:`NEWS_DECAY_WEIGHTS`, so the weights exist in exactly one place."""
    terms = []
    for offset, weight in enumerate(NEWS_DECAY_WEIGHTS):
        value = "s_t" if offset == 0 else f"coalesce(lag(s_t, {offset}) OVER w_order, 0.0)"
        terms.append(f"{weight} * {value}")
    return f"({' + '.join(terms)}) / {NEWS_DECAY_DENOMINATOR}"


def outer_projections() -> tuple[tuple[str, str], ...]:
    """Ordered ``(column, expression)`` pairs for the final SELECT. Order matches the DDL."""
    return (
        ("ticker", "ticker"),
        ("trade_date", "trade_date"),
        ("close", "`close`"),
        ("volume", "volume"),
        ("log_return", "log_return"),
        ("return_5d", "return_5d"),
        ("momentum_5d", momentum_expr()),
        ("realized_vol_20d", realized_vol_expr()),
        ("volume_zscore_20d", volume_zscore_expr()),
        ("s_t", "s_t"),
        ("news_sentiment_3d", news_sentiment_3d_expr()),
        ("news_count", "news_count"),
    )


#: The table's column order, derived from the projections so the two cannot drift.
FEATURE_COLUMNS = tuple(name for name, _ in outer_projections())


def build_features_sql(catalog: str) -> str:
    """The full SELECT that produces one row per (ticker, session).

    Structure: restrict prices to XNYS sessions, aggregate news onto sessions through the
    calendar map, left-join the two, then apply the window functions. The left join is what makes
    ``s_t`` and ``news_count`` 0 for a session with no articles rather than dropping the row.

    The ``coalesce`` inside ``avg`` matters: Spark's ``avg`` skips NULLs, so a missing score would
    be excluded from the mean while still counting in ``news_count``, leaving the two columns
    describing different sets of articles. A-3 already degrades an unscored article to neutral, so
    a NULL here is counted as 0 — which is also what :func:`session_news_aggregates` does.
    """
    prices_fqn = qualified(catalog, PRICES_TABLE)
    news_fqn = qualified(catalog, NEWS_TABLE)
    news_date = session_date_expr("n.published_at")
    projections = ",\n       ".join(
        expression if expression == name else f"{expression} AS {quote_identifier(name)}"
        for name, expression in outer_projections()
    )

    return f"""SELECT {projections}
FROM (
    SELECT ticker,
           trade_date,
           `close`,
           volume,
           s_t,
           news_count,
           {log_return_expr()} AS log_return,
           {return_over_lag_expr()} AS return_5d
    FROM (
        SELECT p.ticker,
               p.trade_date,
               p.`close`,
               p.volume,
               coalesce(nw.s_t, 0.0) AS s_t,
               coalesce(nw.news_count, 0) AS news_count
        FROM {prices_fqn} p
        JOIN {SESSIONS_VIEW} s ON p.trade_date = s.session_date
        LEFT JOIN (
            SELECT n.ticker,
                   m.session_date AS trade_date,
                   avg(coalesce(CAST(n.sentiment_score AS DOUBLE), 0.0)) AS s_t,
                   count(*) AS news_count
            FROM {news_fqn} n
            JOIN {SESSION_MAP_VIEW} m ON {news_date} = m.calendar_date
            GROUP BY n.ticker, m.session_date
        ) nw ON p.ticker = nw.ticker AND p.trade_date = nw.trade_date
    )
    WINDOW w_order AS (PARTITION BY ticker ORDER BY trade_date)
)
WINDOW w_order AS (PARTITION BY ticker ORDER BY trade_date),
       w_momentum AS (PARTITION BY ticker ORDER BY trade_date {_trailing_frame(MOMENTUM_WINDOW)}),
       w_vol AS (PARTITION BY ticker ORDER BY trade_date {_trailing_frame(VOL_WINDOW)}),
       w_volume AS (PARTITION BY ticker ORDER BY trade_date {_trailing_frame(VOLUME_WINDOW)})"""


# --------------------------------------------------------------------- entry point


def _price_date_range(spark: Any, prices_fqn: str) -> tuple[date, date] | None:
    row = spark.sql(
        f"SELECT min(trade_date) AS first_date, max(trade_date) AS last_date FROM {prices_fqn}"
    ).first()
    if row is None or row["first_date"] is None:
        return None
    return row["first_date"], row["last_date"]


def _register_calendar_views(spark: Any, first_date: date, last_date: date) -> int:
    """Publish the session list and the date -> session map as temp views.

    The calendar is read once, on the driver, and the result is a few hundred rows per year. The
    range is the price range: an article that rolls forward past the last bar has no feature row
    to attach to yet, and it will be picked up by the run that ingests that bar.
    """
    sessions = trading_sessions(first_date, last_date)
    session_map = next_session_map(sessions, first_date, last_date)

    session_rows = spark.createDataFrame(
        [(session,) for session in sessions], schema=SESSIONS_SCHEMA_DDL
    )
    session_rows.createOrReplaceTempView(SESSIONS_VIEW)

    map_rows = spark.createDataFrame(
        sorted(session_map.items()), schema=SESSION_MAP_SCHEMA_DDL
    )
    map_rows.createOrReplaceTempView(SESSION_MAP_VIEW)

    log.info(
        "registered %s calendar sessions=%d mapped_dates=%d range=%s..%s",
        CALENDAR_NAME,
        len(sessions),
        len(session_map),
        first_date,
        last_date,
    )
    return len(sessions)


def main(spark: Any, config: Mapping) -> dict:
    """Build ``silver.daily_features``.

    Callable identically from a workflow task and from a notebook cell::

        from src.pipelines import feature_pipeline
        feature_pipeline.main(spark, config)

    Returns a summary dict for notebook display. Exactly one ``bronze.ingestion_runs`` row is
    written per call, on success and on failure, the same as the ingestion tasks.
    """
    catalog = str(config["catalog"])
    prices_fqn = qualified(catalog, PRICES_TABLE)
    news_fqn = qualified(catalog, NEWS_TABLE)
    target_fqn = qualified(catalog, TARGET_TABLE)
    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    sessions = 0

    try:
        for table in (prices_fqn, news_fqn, target_fqn):
            require_table(spark, table)
        pin_session_timezone_to_utc(spark)

        date_range = _price_date_range(spark, prices_fqn)
        if date_range is None:
            log.warning("%s: %s has no rows, nothing to build", TASK_NAME, prices_fqn)
        else:
            sessions = _register_calendar_views(spark, *date_range)
            run.rows_written = merge_select(
                spark, target_fqn, build_features_sql(catalog), MERGE_KEYS
            )
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s complete run_id=%s rows_merged=%d sessions=%d target=%s",
        TASK_NAME,
        run.run_id,
        run.rows_written,
        sessions,
        target_fqn,
    )
    return {
        "task": TASK_NAME,
        "run_id": run.run_id,
        "target": target_fqn,
        "rows_merged": run.rows_written,
        "sessions": sessions,
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
