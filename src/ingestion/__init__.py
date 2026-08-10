"""Massive API ingestion into the bronze layer.

Owns the only outbound network calls to the market-data vendor and the only writes to
``bronze.prices_raw``, ``bronze.news_raw`` and ``bronze.ingestion_runs``.

Because it owns those writes, this package module also holds the shared bronze write layer that
``ingest_prices`` and ``ingest_news`` both use: :func:`merge_rows`, :func:`record_run`,
:func:`max_value` and the pure helpers around them. The frozen repo layout (spec B0) fixes
``src/ingestion/`` at ``massive_client.py | ingest_prices.py | ingest_news.py``, so a fourth
module is not an option, and duplicating a MERGE across two jobs is how the two copies drift.

Two deliberate implementation choices:

- Nothing here imports ``pyspark``. Every Spark interaction goes through the injected
  ``spark`` session plus DDL-string schemas, so the pure functions in this package stay
  importable — and unit-testable — on a machine with no Spark installed.
- Every write is a MERGE on declared keys (spec rule 6). The daily workflow retries ingestion
  tasks, so a retry must update rows rather than duplicate them.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

#: Status values written to ``bronze.ingestion_runs.status``.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

RUNS_TABLE = "bronze.ingestion_runs"
RUNS_MERGE_KEYS = ("run_id",)
RUNS_COLUMNS = (
    "run_id",
    "task",
    "started_at",
    "finished_at",
    "status",
    "rows_written",
    "error",
)
RUNS_SCHEMA_DDL = (
    "run_id STRING, task STRING, started_at TIMESTAMP, finished_at TIMESTAMP, "
    "status STRING, rows_written BIGINT, error STRING"
)

#: Truncation limit for the ledger's ``error`` column — a summary, not a stack trace.
MAX_ERROR_CHARS = 1000


@dataclass
class RunRecord:
    """One row of ``bronze.ingestion_runs``."""

    run_id: str
    task: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = STATUS_SUCCEEDED
    rows_written: int = 0
    error: str | None = None

    def as_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "rows_written": int(self.rows_written),
            "error": self.error,
        }


def utc_now() -> datetime:
    """Timezone-aware UTC now. Never a naive timestamp (architecture doc §5)."""
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return str(uuid.uuid4())


# ------------------------------------------------------------------ pure helpers


def resolve_universe(config: Mapping, watchlist: Iterable[str] | None = None) -> list[str]:
    """Ingestion universe: the seed tickers plus every watchlist ticker (spec A1).

    Deduplicated and sorted so a run's request order is deterministic. ``watchlist`` is a
    parameter rather than a Lakebase read because Lakebase does not exist until C-2; the daily
    job will pass ``src.database.lakebase.get_watchlist()`` once it does.
    TODO (C-2): wire the watchlist read into the workflow task entry point.
    """
    seed = (config.get("tickers") or {}).get("seed") or []
    tickers = {str(t).strip().upper() for t in seed if str(t).strip()}
    tickers |= {str(t).strip().upper() for t in (watchlist or []) if str(t).strip()}
    return sorted(tickers)


def dedupe_by_keys(rows: Sequence[Mapping], keys: Sequence[str]) -> list[dict]:
    """Collapse rows sharing a MERGE key, last occurrence winning.

    Required, not defensive: Delta fails a MERGE outright when several source rows match the
    same target row, and a paginated fetch can legitimately return the same article twice.
    """
    collapsed: dict[tuple, dict] = {}
    for row in rows:
        collapsed[tuple(row.get(k) for k in keys)] = dict(row)
    return list(collapsed.values())


def rows_to_tuples(rows: Sequence[Mapping], columns: Sequence[str]) -> list[tuple]:
    """Project dict rows into positional tuples matching ``columns``.

    ``createDataFrame`` is given tuples plus an explicit DDL schema rather than dicts, so a
    column added to the table but forgotten in the row builder fails loudly at write time.
    """
    return [tuple(row.get(column) for column in columns) for row in rows]


def truncate_error(message: str) -> str:
    return message if len(message) <= MAX_ERROR_CHARS else message[: MAX_ERROR_CHARS - 3] + "..."


# ----------------------------------------------------------------- Spark boundary


def qualified(catalog: str, table: str) -> str:
    """``market_intel`` + ``bronze.prices_raw`` -> ``market_intel.bronze.prices_raw``."""
    return f"{catalog}.{table}"


def require_table(spark: Any, fqn: str) -> None:
    """Fail with an actionable message when the target table was never created."""
    if not spark.catalog.tableExists(fqn):
        raise RuntimeError(
            f"Table {fqn} does not exist. Run setup/create_catalog.sql then "
            "setup/create_delta_tables.sql before the ingestion tasks."
        )


def max_value(spark: Any, fqn: str, column: str, filter_column: str, filter_value: Any) -> Any:
    """``SELECT max(column) FROM fqn WHERE filter_column = filter_value`` (None when empty).

    This is the per-ticker watermark read. Parameter markers keep the ticker out of the SQL text
    (requires DBR 14.1+ / Spark 3.5+, which the workspace runtime satisfies).
    """
    row = spark.sql(
        f"SELECT max({column}) AS watermark FROM {fqn} WHERE {filter_column} = :value",
        args={"value": filter_value},
    ).first()
    return None if row is None else row["watermark"]


def merge_rows(
    spark: Any,
    fqn: str,
    rows: Sequence[Mapping],
    *,
    columns: Sequence[str],
    schema_ddl: str,
    keys: Sequence[str],
) -> int:
    """MERGE ``rows`` into ``fqn`` on ``keys``. Returns the number of rows merged.

    Never a blind INSERT (spec rule 6): re-running a task updates matched rows in place, so the
    workflow's automatic retries cannot duplicate data.

    The returned count is the post-dedupe staged row count — the rows this run presented to the
    MERGE — which is what ``bronze.ingestion_runs.rows_written`` records. Splitting it into
    inserted-versus-updated would mean reading Delta's operation metrics, which the spec does not
    ask for.
    """
    if not rows:
        return 0

    staged = dedupe_by_keys(rows, keys)
    view = f"_stage_{uuid.uuid4().hex}"
    frame = spark.createDataFrame(rows_to_tuples(staged, columns), schema=schema_ddl)
    frame.createOrReplaceTempView(view)
    try:
        condition = " AND ".join(f"t.{key} = s.{key}" for key in keys)
        spark.sql(
            f"MERGE INTO {fqn} AS t USING {view} AS s ON {condition} "
            "WHEN MATCHED THEN UPDATE SET * "
            "WHEN NOT MATCHED THEN INSERT *"
        )
    finally:
        spark.catalog.dropTempView(view)

    log.info("merged rows table=%s rows=%d keys=%s", fqn, len(staged), ",".join(keys))
    return len(staged)


def record_run(spark: Any, catalog: str, run: RunRecord) -> None:
    """Write the run's ledger row. Called on success AND on failure (spec A-2).

    Ledger failures are logged, never raised: losing an audit row must not turn a successful
    ingestion into a failed task, and must not mask the original exception when one is in flight.
    """
    fqn = qualified(catalog, RUNS_TABLE)
    try:
        merge_rows(
            spark,
            fqn,
            [run.as_row()],
            columns=RUNS_COLUMNS,
            schema_ddl=RUNS_SCHEMA_DDL,
            keys=RUNS_MERGE_KEYS,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.error(
            "failed to write ingestion_runs row task=%s run_id=%s error=%s",
            run.task,
            run.run_id,
            type(exc).__name__,
        )
