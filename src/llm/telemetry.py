"""Lightweight model-call telemetry (spec C-3).

Appends one record per model call to the Delta table ``gold.model_calls``::

    {ts, task, model, latency_ms, ok, in_tokens, out_tokens}

Token counts are recorded when the endpoint reports them.

This is instrumentation, not a separate observability project. No AI-routing dashboard.

BUFFERED, THEN MERGED. :func:`record` only appends to an in-memory buffer — it is called from
inside a chat turn and must not put a Spark write on the interactive path. :func:`flush` drains
the buffer through the shared write layer, which means a MERGE on declared keys (rule 4) rather
than an append: a retried flush must not duplicate rows.

That MERGE needs an identity, and ``{ts, task, model, ...}`` has none, so each record also
carries a ``call_id`` (uuid4) and that is the MERGE key. The spec's column list says what to
record; a surrogate key is what makes recording it idempotent.

THREE MODES, because the app cannot run Spark. ``telemetry.mode`` in ``config/config.yaml``,
overridable by the ``TELEMETRY_MODE`` environment variable:

- ``delta`` — MERGE the buffer into ``gold.model_calls``. Needs an injected SparkSession.
- ``log``   — keep the records and log them on flush. The Databricks App has no SparkSession
  (spec C-5), so this is what the agent page runs with; app.yaml sets it.
- ``off``   — record nothing.

The write goes through ``src.ingestion.merge_rows`` — the same function the bronze, silver and
gold writes use. Nothing here imports ``pyspark``: the session is injected, exactly as it is in
``src/ingestion/`` and ``src/pipelines/``, so this module stays importable inside the app.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.ingestion import merge_rows, qualified, require_table
from src.llm import config_section

__all__ = [
    "MAX_BUFFERED_RECORDS",
    "MODEL_CALLS_COLUMNS",
    "MODEL_CALLS_KEYS",
    "MODEL_CALLS_SCHEMA_DDL",
    "MODEL_CALLS_TABLE",
    "MODES",
    "MODE_DELTA",
    "MODE_LOG",
    "MODE_OFF",
    "ModelCall",
    "TelemetryModeError",
    "buffered",
    "clear",
    "configure",
    "flush",
    "record",
    "resolve_mode",
]

log = logging.getLogger(__name__)

MODEL_CALLS_TABLE = "gold.model_calls"
MODEL_CALLS_COLUMNS = (
    "call_id",
    "ts",
    "task",
    "model",
    "latency_ms",
    "ok",
    "in_tokens",
    "out_tokens",
)
MODEL_CALLS_SCHEMA_DDL = (
    "call_id STRING, ts TIMESTAMP, task STRING, model STRING, latency_ms DOUBLE, "
    "ok BOOLEAN, in_tokens BIGINT, out_tokens BIGINT"
)
MODEL_CALLS_KEYS = ("call_id",)

MODE_DELTA = "delta"
MODE_LOG = "log"
MODE_OFF = "off"
MODES = (MODE_DELTA, MODE_LOG, MODE_OFF)

#: Environment override, read before the config file. app.yaml sets it to "log".
MODE_ENV_VAR = "TELEMETRY_MODE"

#: Mode used when neither the environment nor the config says anything. Log-only rather than
#: delta: a default that needs a SparkSession would turn "someone forgot to configure telemetry"
#: into a failure inside the agent.
DEFAULT_MODE = MODE_LOG

#: Buffer cap. The app is a long-lived process whose flush is best-effort, so the buffer is a
#: bounded deque: dropping the oldest telemetry is survivable, growing without limit is not.
MAX_BUFFERED_RECORDS = 1000


class TelemetryModeError(ValueError):
    """The configured telemetry mode is not one of :data:`MODES`."""


@dataclass(frozen=True)
class ModelCall:
    """One row of ``gold.model_calls``."""

    ts: datetime
    task: str
    model: str
    latency_ms: float
    ok: bool
    in_tokens: int | None = None
    out_tokens: int | None = None
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_row(self) -> dict:
        return {
            "call_id": self.call_id,
            "ts": self.ts,
            "task": self.task,
            "model": self.model,
            "latency_ms": float(self.latency_ms),
            "ok": bool(self.ok),
            "in_tokens": None if self.in_tokens is None else int(self.in_tokens),
            "out_tokens": None if self.out_tokens is None else int(self.out_tokens),
        }


_lock = threading.Lock()
_buffer: deque[ModelCall] = deque(maxlen=MAX_BUFFERED_RECORDS)
_mode_override: str | None = None
_dropped = 0


def configure(mode: str | None) -> None:
    """Pin the telemetry mode, or restore config/environment resolution with ``None``."""
    global _mode_override

    with _lock:
        _mode_override = None if mode is None else _validate_mode(mode, source="configure()")


def resolve_mode(config: Mapping[str, Any] | None = None) -> str:
    """:func:`configure` > ``TELEMETRY_MODE`` > ``telemetry.mode`` > :data:`DEFAULT_MODE`.

    The environment wins over the file because one repository config is read by both the jobs and
    the app, and the app is the deployment that cannot run Spark.
    """
    if _mode_override is not None:
        return _mode_override

    env_mode = os.environ.get(MODE_ENV_VAR, "").strip()
    if env_mode:
        return _validate_mode(env_mode, source=MODE_ENV_VAR)

    configured = config_section("telemetry", config).get("mode")
    if configured:
        return _validate_mode(str(configured), source="config telemetry.mode")

    return DEFAULT_MODE


def record(
    *,
    task: str,
    model: str,
    latency_ms: float,
    ok: bool,
    in_tokens: int | None = None,
    out_tokens: int | None = None,
    ts: datetime | None = None,
    config: Mapping[str, Any] | None = None,
) -> ModelCall | None:
    """Buffer one model call. Returns the record, or ``None`` in ``off`` mode.

    Failed calls are recorded too, with ``ok=False`` and no token counts — a telemetry table that
    only holds successes cannot answer the one question it exists for.
    """
    global _dropped

    if resolve_mode(config) == MODE_OFF:
        return None

    call = ModelCall(
        ts=ts or datetime.now(timezone.utc),
        task=task,
        model=model,
        latency_ms=float(latency_ms),
        ok=bool(ok),
        in_tokens=in_tokens,
        out_tokens=out_tokens,
    )

    with _lock:
        if len(_buffer) == MAX_BUFFERED_RECORDS:
            _dropped += 1
            if _dropped == 1 or _dropped % 100 == 0:
                log.warning(
                    "model telemetry buffer full (%d records); dropping oldest, dropped=%d",
                    MAX_BUFFERED_RECORDS,
                    _dropped,
                )
        _buffer.append(call)

    return call


def buffered() -> tuple[ModelCall, ...]:
    """A snapshot of the buffer, for the flush job and for the C-D telemetry mini-view."""
    with _lock:
        return tuple(_buffer)


def clear() -> None:
    """Drop everything buffered without writing it."""
    global _dropped

    with _lock:
        _buffer.clear()
        _dropped = 0


def flush(
    spark: Any = None,
    *,
    catalog: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> int:
    """Drain the buffer and return how many records it held.

    In ``delta`` mode the records are MERGEd into ``{catalog}.gold.model_calls``, which needs
    ``spark``; in ``log`` mode they are logged and nothing is written; in ``off`` mode there is
    nothing to drain. ``catalog`` defaults to the config's ``catalog`` key.

    A failed write puts the records BACK in the buffer and re-raises, so the next flush retries
    them. That is safe precisely because the write is a MERGE on ``call_id``: a partially applied
    flush cannot duplicate a row.
    """
    mode = resolve_mode(config)

    with _lock:
        pending = list(_buffer)
        _buffer.clear()

    if not pending:
        return 0

    if mode == MODE_OFF:
        return 0

    if mode == MODE_LOG:
        for call in pending:
            log.info(
                "model call telemetry ts=%s task=%s model=%s latency_ms=%.0f ok=%s "
                "in_tokens=%s out_tokens=%s",
                call.ts.isoformat(),
                call.task,
                call.model,
                call.latency_ms,
                call.ok,
                call.in_tokens,
                call.out_tokens,
            )
        log.info("model telemetry flushed mode=log records=%d written=0", len(pending))
        return len(pending)

    if spark is None:
        _restore(pending)
        raise TelemetryModeError(
            "telemetry mode is 'delta' but no SparkSession was passed to flush(). The Databricks "
            f"App has no SparkSession — set {MODE_ENV_VAR}=log there (app.yaml does)."
        )

    target = catalog or _catalog_from_config(config)
    fqn = qualified(target, MODEL_CALLS_TABLE)
    try:
        require_table(spark, fqn)
        written = merge_rows(
            spark,
            fqn,
            [call.as_row() for call in pending],
            columns=MODEL_CALLS_COLUMNS,
            schema_ddl=MODEL_CALLS_SCHEMA_DDL,
            keys=MODEL_CALLS_KEYS,
        )
    except BaseException:
        _restore(pending)
        raise

    log.info("model telemetry flushed mode=delta table=%s records=%d", fqn, written)
    return written


# ------------------------------------------------------------------------- internals


def _restore(pending: Sequence[ModelCall]) -> None:
    """Put drained records back at the front of the buffer, oldest first."""
    with _lock:
        _buffer.extendleft(reversed(pending))


def _catalog_from_config(config: Mapping[str, Any] | None) -> str:
    from src.llm import load_config

    source = config if config is not None else load_config()
    catalog = source.get("catalog")
    if not catalog:
        raise TelemetryModeError("config key 'catalog' is not set, so gold.model_calls has no home")
    return str(catalog)


def _validate_mode(mode: str, *, source: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in MODES:
        raise TelemetryModeError(f"{source} is {mode!r}, expected one of {list(MODES)}")
    return normalized
