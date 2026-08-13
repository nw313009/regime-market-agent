"""``silver.news_recent``: the rolling news window the app reads (C-5), maintained by C-6.

Two tasks live here, and they share one rule about where the window ends.

- :func:`refresh` — the daily task. MERGE the last ``news_recent.window_days`` of
  ``silver.news_articles``, then delete whatever now sits below the retention floor.
- :func:`backfill` — the optional task. Extend the window BACKWARD by ``news_recent.batch_days``
  per run until it reaches ``news_recent.backfill_floor``, then do nothing at all.

THE ONE RULE THEY SHARE. Run naively, these two fight: the refresh deletes everything older than
90 days, which is exactly what the backfill just added, and the pair burns a warehouse rewriting
the same rows forever. :func:`retention_floor` is the single place that resolves it, from a single
config key — ``news_recent.include_backfill``. With the backfill off, the floor is
``today - window_days`` and the table is a strict rolling window. With it on, the floor is
``backfill_floor`` and the refresh stops deleting history the backfill is there to build. The
workflow reads the same key to decide whether to schedule the backfill task at all, so the two
halves of the decision cannot drift apart.

WHY A TABLE AND NOT A VIEW: see the DDL comment. The page runs this query on every rerun over a
serverless warehouse; a view would re-scan the growing archive each time.

IT IS ALSO THE INDEXED CORPUS (C-1). ``search.source_table`` points at this table, so the window
is what the agent can retrieve and rows aging out here disappear from the index on the next sync.
Two consequences live in this module: :data:`COLUMNS` carries ``doc_id`` and ``embedding_text``,
which the page never displays, and :func:`unchanged_predicate` keeps the daily refresh from
rewriting rows that did not change — a rewritten row is a Change Data Feed event, and a feed event
is an embedding call.

DATES ARE FORMATTED, NOT INTERPOLATED. The MERGE source is SQL text, so there is no parameter
marker to bind — :func:`_date_literal` therefore takes a ``datetime.date`` and refuses anything
else, which makes the value unforgeable by construction rather than by review. Every date here
comes from the clock or from config; none comes from a user.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from src.pipelines import (
    STATUS_FAILED,
    RunRecord,
    merge_select,
    new_run_id,
    qualified,
    quote_identifier,
    record_run,
    require_table,
    truncate_error,
    utc_now,
)

__all__ = [
    "COLUMNS",
    "MERGE_KEYS",
    "SOURCE_TABLE",
    "TARGET_TABLE",
    "TASK_BACKFILL",
    "TASK_REFRESH",
    "WindowSettings",
    "backfill",
    "next_backfill_window",
    "refresh",
    "retention_floor",
    "settings_from_config",
    "source_sql",
    "unchanged_predicate",
    "window_start",
]

log = logging.getLogger(__name__)

TASK_REFRESH = "refresh_news_recent"
TASK_BACKFILL = "backfill_news_recent"

SOURCE_TABLE = "silver.news_articles"
TARGET_TABLE = "silver.news_recent"

#: Exactly the target's columns, in DDL order. ``MERGE ... UPDATE SET * / INSERT *`` matches by
#: NAME, so a column missing from this projection fails the write rather than defaulting to NULL.
#:
#: ``embedding_text`` and ``doc_id`` are here for the AI Search index, not for the page: the index
#: is built on this table (C-1), and a Delta Sync index needs its embedding source column and its
#: single primary key column present in the source.
COLUMNS: tuple[str, ...] = (
    "article_id",
    "ticker",
    "published_at",
    "title",
    "publisher",
    "sentiment_label",
    "sentiment_score",
    "article_url",
    "embedding_text",
    "doc_id",
)

MERGE_KEYS: tuple[str, ...] = ("article_id", "ticker")

#: The window column. Retention, refresh and backfill are all expressed against it.
WINDOW_COLUMN = "published_at"


@dataclass(frozen=True)
class WindowSettings:
    """The three numbers that define the window, plus the flag that resolves the conflict."""

    window_days: int
    batch_days: int
    backfill_floor: date
    include_backfill: bool = False


def settings_from_config(config: Mapping) -> WindowSettings:
    """Read the ``news_recent`` block, with the same defaults the shipped config carries."""
    section = dict(config.get("news_recent") or {})
    return WindowSettings(
        window_days=int(section.get("window_days", 90)),
        batch_days=int(section.get("batch_days", 90)),
        backfill_floor=_as_date(section.get("backfill_floor", "2024-08-01")),
        include_backfill=bool(section.get("include_backfill", False)),
    )


def window_start(today: date, settings: WindowSettings) -> date:
    """The first day of the rolling window the daily refresh republishes."""
    return today - timedelta(days=settings.window_days)


def retention_floor(today: date, settings: WindowSettings) -> date:
    """The oldest day :func:`refresh` will KEEP. Everything strictly older is deleted.

    The single place the refresh-versus-backfill conflict is resolved; see the module docstring.
    Never below ``backfill_floor``, so a misconfigured ``window_days`` cannot make the daily task
    quietly delete the history the backfill spent runs assembling.
    """
    rolling = window_start(today, settings)
    if not settings.include_backfill:
        return rolling
    return min(rolling, settings.backfill_floor)


def next_backfill_window(
    oldest_held: date | None,
    today: date,
    settings: WindowSettings,
) -> tuple[date, date] | None:
    """The next ``[start, end)`` slice for :func:`backfill`, or ``None`` when it is finished.

    ``oldest_held`` is the oldest ``published_at`` currently in the table — how deep the window
    already goes. ``None`` means the table is empty, in which case the backfill starts from the
    rolling window's edge, the same place the refresh would have filled to.

    The end is EXCLUSIVE and equals ``oldest_held``, so the boundary day is not re-merged: it is
    already there, and re-reading it every run would make the task's cost grow with its progress.
    """
    end = oldest_held or window_start(today, settings)
    if end <= settings.backfill_floor:
        return None
    start = max(settings.backfill_floor, end - timedelta(days=settings.batch_days))
    return start, end


def unchanged_predicate(catalog: str, alias: str = "a", target: str = "held") -> str:
    """``NOT EXISTS (...)``: true for a source row the target does not already hold verbatim.

    THIS IS WHAT STOPS THE INDEX RE-EMBEDDING ITSELF EVERY NIGHT. The daily refresh re-presents the
    whole 90-day window, and ``WHEN MATCHED THEN UPDATE SET *`` rewrites every matched row whether
    or not a value differs. Delta records those rewrites in the Change Data Feed, the AI Search
    Delta Sync index (C-1) reads that feed, and managed embeddings re-embed whatever the feed says
    changed — so the naive refresh would pay to re-embed several thousand unchanged articles daily
    and leave the index resyncing for minutes each run.

    Comparison is ``<=>`` (null-safe) on EVERY non-key column, not just the embedding source, so a
    corrected sentiment label or a fixed title still propagates. Only rows where nothing at all
    moved are skipped. The anti-join costs a join of the window against itself; the alternative
    costs an embedding call per row per day.

    The target alias is ``held``, not ``t``: this text is embedded in a ``MERGE ... AS t``, and an
    inner ``t`` would shadow the target alias for anyone reading — or editing — the statement.
    """
    source_fqn = qualified(catalog, TARGET_TABLE)
    keys = " AND ".join(
        f"{target}.{quote_identifier(key)} = {alias}.{quote_identifier(key)}" for key in MERGE_KEYS
    )
    payload = "".join(
        f"\n         AND {target}.{quote_identifier(column)} <=> {alias}.{quote_identifier(column)}"
        for column in COLUMNS
        if column not in MERGE_KEYS
    )
    return (
        f"NOT EXISTS (SELECT 1 FROM {source_fqn} AS {target}\n"
        f"       WHERE {keys}{payload})"
    )


def source_sql(catalog: str, start: date, end: date | None = None) -> str:
    """The projection of ``silver.news_articles`` for one window, as MERGE source text.

    ``end`` is exclusive and optional: the daily refresh has no upper bound, since "the last 90
    days" includes anything published this morning.

    Rows with a NULL ``published_at`` are excluded by the comparison itself and that is correct —
    an article with no publication time cannot be placed in a window, and the app's list is
    ordered by that column.

    Rows the target already holds unchanged are excluded too; see :func:`unchanged_predicate`. The
    consequence for the ledger is that ``rows_written`` counts what actually moved, not the size of
    the window — which is the more useful number anyway, and the one a "nothing happened today" run
    reports as zero.
    """
    alias = "a"
    projection = ",\n       ".join(f"{alias}.{quote_identifier(column)}" for column in COLUMNS)
    bounds = f"{alias}.{WINDOW_COLUMN} >= {_date_literal(start)}"
    if end is not None:
        bounds += f"\n  AND {alias}.{WINDOW_COLUMN} < {_date_literal(end)}"
    return (
        f"SELECT {projection}\n"
        f"FROM {qualified(catalog, SOURCE_TABLE)} AS {alias}\n"
        f"WHERE {bounds}\n"
        f"  AND {unchanged_predicate(catalog, alias=alias)}"
    )


def refresh(
    spark: Any,
    config: Mapping,
    *,
    today: date | None = None,
    catalog: str | None = None,
) -> dict:
    """Daily ``refresh_news_recent``: republish the window, then trim what fell out of it.

    MERGE first and DELETE second, deliberately. The other order leaves the app with a hole in its
    news list for the length of the merge, and the page has no way to tell an empty window from a
    momentarily empty table.
    """
    catalog = catalog or str(config["catalog"])
    settings = settings_from_config(config)
    today = today or utc_now().date()

    target_fqn = qualified(catalog, TARGET_TABLE)
    start = window_start(today, settings)
    floor = retention_floor(today, settings)

    run = RunRecord(run_id=new_run_id(), task=TASK_REFRESH, started_at=utc_now())
    deleted_below: date | None = None

    try:
        require_table(spark, qualified(catalog, SOURCE_TABLE))
        require_table(spark, target_fqn)

        run.rows_written = merge_select(
            spark, target_fqn, source_sql(catalog, start), MERGE_KEYS
        )
        spark.sql(
            f"DELETE FROM {target_fqn} WHERE {WINDOW_COLUMN} < {_date_literal(floor)}"
        )
        deleted_below = floor
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s run_id=%s merged=%d window=%s.. retained_from=%s",
        TASK_REFRESH,
        run.run_id,
        run.rows_written,
        start,
        floor,
    )
    return {
        "task": TASK_REFRESH,
        "run_id": run.run_id,
        "window_start": start,
        "retention_floor": deleted_below,
        "rows_merged": run.rows_written,
    }


def backfill(
    spark: Any,
    config: Mapping,
    *,
    today: date | None = None,
    catalog: str | None = None,
) -> dict:
    """Optional ``backfill_news_recent``: one batch older, or nothing left to do.

    Progressive by design. One run moves the window back by ``batch_days`` and stops; the next run
    picks up where it left off, reading its position from the table rather than from a stored
    cursor. When the floor is reached the task becomes a permanent no-op that still writes its
    ledger row, so a scheduled job does not start failing the day the backfill finishes.
    """
    catalog = catalog or str(config["catalog"])
    settings = settings_from_config(config)
    today = today or utc_now().date()

    target_fqn = qualified(catalog, TARGET_TABLE)
    run = RunRecord(run_id=new_run_id(), task=TASK_BACKFILL, started_at=utc_now())
    window: tuple[date, date] | None = None

    try:
        require_table(spark, qualified(catalog, SOURCE_TABLE))
        require_table(spark, target_fqn)

        oldest = oldest_held(spark, target_fqn)
        window = next_backfill_window(oldest, today, settings)
        if window is not None:
            run.rows_written = merge_select(
                spark, target_fqn, source_sql(catalog, *window), MERGE_KEYS
            )
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    if window is None:
        log.info("%s run_id=%s complete: floor %s reached", TASK_BACKFILL, run.run_id, settings.backfill_floor)
    else:
        log.info(
            "%s run_id=%s merged=%d window=%s..%s floor=%s",
            TASK_BACKFILL,
            run.run_id,
            run.rows_written,
            window[0],
            window[1],
            settings.backfill_floor,
        )
    return {
        "task": TASK_BACKFILL,
        "run_id": run.run_id,
        "done": window is None,
        "window": None if window is None else {"start": window[0], "end": window[1]},
        "rows_merged": run.rows_written,
    }


def oldest_held(spark: Any, target_fqn: str) -> date | None:
    """How deep the window currently goes, or ``None`` when the table is empty.

    The backfill's cursor. Read from the data rather than kept in a state table on purpose: a
    stored cursor and the rows it describes are two things that can disagree, and the recovery
    from that disagreement is to recompute the cursor from the data anyway.
    """
    row = spark.sql(f"SELECT CAST(MIN({WINDOW_COLUMN}) AS DATE) AS oldest FROM {target_fqn}").first()
    if row is None:
        return None
    value = row["oldest"]
    return None if value is None else _as_date(value)


def _date_literal(value: date) -> str:
    """A SQL DATE literal from a ``datetime.date``, and from nothing else.

    The type check is the security control: MERGE source text has no parameter markers, so this is
    the only thing standing between a date and a string that happens to contain SQL.
    """
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"expected a datetime.date, got {type(value).__name__}: {value!r}")
    return f"DATE '{value.isoformat()}'"


def _as_date(value: Any) -> date:
    """Normalize an ISO string, a ``datetime`` or a ``date`` to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip())
    converted = getattr(value, "to_pydatetime", None)
    if converted is not None:
        return converted().date()
    raise TypeError(f"expected a date, got {type(value).__name__}: {value!r}")
