"""``sync_lakebase_history``: Lakebase (Postgres) rows captured into Delta (spec A3, C-6).

This is the CDC direction of the demo. Operational writes — a watchlist add, a saved report — land
in Postgres through the app; this task carries them into ``market_intel.gold.lb_*_history`` so the
lakehouse holds the history of what the application did. Analytical data flows the other way and
the two never loop (spec A3).

WHY NOT psycopg. The obvious implementation reuses ``src/database/lakebase.py``, and it cannot:
psycopg 3.3.4 SIGABRTs a serverless kernel inside libpq AT IMPORT TIME, exit code 134, before any
of our code runs (see the note in ``requirements-databricks.txt``). A workflow task that imports it
does not fail, it dies. So this module never imports that one, and
``tests/test_import_boundaries.py`` enforces it. The four connection facts are read from config and
env instead — duplicating four strings is the cheap side of that trade.

THE MECHANISM, verified against the installed databricks-sdk 0.125.0 rather than recalled:

1. ``WorkspaceClient().postgres.generate_database_credential(endpoint=...)`` returns a
   ``DatabaseCredential`` with a ``token`` field — an OAuth token usable as a Postgres password.
   This is a plain HTTPS call; nothing in the SDK imports a Postgres driver.
2. The SDK offers NO Spark integration for Postgres — ``PostgresAPI`` manages infrastructure and
   mints credentials, and its own docstring points at "direct SQL connections" for data. So the
   read is plain Spark JDBC (``spark.read.format("jdbc")``) with the PostgreSQL driver that
   Databricks Runtime ships, the minted token as the password, and ``sslmode=require``.
3. The token is short-lived and minted per run, so nothing is stored and rule 5 holds.

IF JDBC IS BLOCKED IN YOUR WORKSPACE — serverless egress rules or a missing driver — the fallback
is the APP-SIDE SYNC, and this module is shaped so that swapping to it changes one function. The
transport is the injected ``read`` callable; everything after it (:func:`history_rows`, the
watermark read, the MERGE) is transport-agnostic and takes plain dicts. The app-side version reads
the same two tables through ``src/database/lakebase.py`` — psycopg works in the app container,
where it is already proven — and writes through ``src/database/delta.py`` over the SQL warehouse,
which executes a MERGE as happily as a SELECT. It would run on the app's startup path or behind a
button on the agent page rather than in the job, which costs the daily cadence and keeps the
capability. It is deliberately NOT implemented as a second live code path: two syncs writing the
same two tables on different schedules is a worse failure than one sync that has to be triggered
from the app.

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
import os
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
    "SECRET_KEYS",
    "SECRET_SCOPE",
    "TASK_NAME",
    "HistoryTable",
    "LakebaseConnection",
    "access_token",
    "connection_from_config",
    "env_from_secrets",
    "history_rows",
    "jdbc_options",
    "main",
    "read_source",
    "source_query",
    "sync_table",
    "watermark",
]

log = logging.getLogger(__name__)

TASK_NAME = "sync_lakebase_history"

#: The JDBC driver Databricks Runtime ships. Named explicitly so a missing driver fails with the
#: class name in the message rather than as "no suitable driver found for jdbc:postgresql".
JDBC_DRIVER = "org.postgresql.Driver"

#: Where the host and the role come from when they are kept out of the repository. The scope is the
#: one A-0 created for the Massive key; these two keys are added alongside it.
SECRET_SCOPE = "capstone"

#: secret key -> the env name it stands in for, so one mechanism reads both. A serverless notebook
#: task cannot be given environment variables, which is what makes the secret scope the only way to
#: supply these without committing them.
SECRET_KEYS: Mapping[str, str] = {
    "lakebase_host": "PGHOST",
    "lakebase_user": "PGUSER",
}


@dataclass(frozen=True)
class LakebaseConnection:
    """The non-secret facts needed to reach Lakebase. The password is never one of them."""

    host: str
    database: str
    user: str
    endpoint: str
    port: int = 5432
    schema: str = "market_system"

    def jdbc_url(self) -> str:
        """``sslmode=require`` is not optional: this connection crosses the public internet."""
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.database}?sslmode=require"


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


def connection_from_config(
    config: Mapping,
    env: Mapping[str, str] | None = None,
) -> LakebaseConnection:
    """Connection facts from the ``lakebase`` config block, with env vars taking precedence.

    Env first because a job's cluster environment is where a workspace-specific host belongs, and
    config as the checked-in default so a fresh clone has something to correct rather than a
    KeyError to debug. The names match the app's (``PGHOST`` and friends), so one Lakebase project
    is described the same way wherever it is read from.
    """
    env = os.environ if env is None else env
    section = dict(config.get("lakebase") or {})

    def pick(key: str, *env_names: str, default: str = "") -> str:
        for name in env_names:
            value = env.get(name)
            if value:
                return str(value)
        value = section.get(key)
        return str(value) if value else default

    connection = LakebaseConnection(
        host=pick("host", "PGHOST", "LAKEBASE_HOST"),
        database=pick("database", "PGDATABASE", default="databricks_postgres"),
        user=pick("user", "PGUSER"),
        endpoint=pick("endpoint", "LAKEBASE_ENDPOINT"),
        port=int(pick("port", "PGPORT", default="5432")),
        schema=pick("schema", "PGSCHEMA", default="market_system"),
    )

    missing = [
        name
        for name, value in (
            ("host", connection.host),
            ("user", connection.user),
            ("endpoint", connection.endpoint),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"{TASK_NAME}: lakebase {', '.join(missing)} not configured. Put them in the "
            f"`{SECRET_SCOPE}` secret scope as "
            f"{' / '.join(sorted(SECRET_KEYS))} (see env_from_secrets), or in "
            "config/config.yaml under `lakebase:`, or as PGHOST / PGUSER / LAKEBASE_ENDPOINT in "
            "the environment."
        )
    return connection


def env_from_secrets(
    secret_getter: Callable[[str, str], str],
    scope: str = SECRET_SCOPE,
    keys: Mapping[str, str] = SECRET_KEYS,
) -> dict[str, str]:
    """Read the connection facts from a secret scope, shaped as the ``env`` mapping above takes.

    WHY A SECRET SCOPE FOR VALUES THAT ARE NOT SECRETS. The host and the Postgres role are not
    credentials — the password is an OAuth token minted per run and never stored — but they are
    workspace infrastructure, and this repository is public. A serverless notebook task cannot be
    handed environment variables, so the choice is: commit them, or read them from the scope that
    A-0 already created for the Massive key. This is that second option.

    ``secret_getter`` is ``dbutils.secrets.get``, passed in rather than imported, because dbutils
    exists only inside a workspace and this module has to stay importable and testable outside one.

    A MISSING SECRET IS NOT AN ERROR HERE. It is skipped, and the value falls through to config or
    to a real environment variable; if nothing supplies it, connection_from_config raises with a
    message naming all three places to look. Failing in this function instead would replace that
    complete diagnosis with "no such secret", which is only one of the three answers.
    """
    resolved: dict[str, str] = {}
    for secret_key, env_name in keys.items():
        try:
            value = secret_getter(scope, secret_key)
        except Exception as exc:  # noqa: BLE001 — dbutils raises workspace-specific types
            log.info(
                "secret %s/%s unavailable (%s); falling back to config",
                scope,
                secret_key,
                type(exc).__name__,
            )
            continue
        if value:
            resolved[env_name] = str(value)
    return resolved


def access_token(connection: LakebaseConnection, workspace_client: Any = None) -> str:
    """Mint a short-lived Postgres password through the Databricks SDK.

    The import is local so this module stays importable — and testable — on a machine with no SDK
    configuration, which is also how the fake-backed tests avoid a workspace.
    """
    client = workspace_client
    if client is None:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()

    credential = client.postgres.generate_database_credential(endpoint=connection.endpoint)
    token = getattr(credential, "token", None)
    if not token:
        raise RuntimeError(
            f"{TASK_NAME}: generate_database_credential returned no token for endpoint "
            f"{connection.endpoint!r}"
        )
    return str(token)


def jdbc_options(connection: LakebaseConnection, token: str) -> dict[str, str]:
    """The ``spark.read.format("jdbc")`` options. The token is a value here, never a URL fragment."""
    return {
        "url": connection.jdbc_url(),
        "driver": JDBC_DRIVER,
        "user": connection.user,
        "password": token,
    }


def source_query(table: HistoryTable, schema: str, since: datetime | None) -> str:
    """The Postgres SELECT, with the watermark pushed down to the database.

    Pushed down rather than filtered in Spark because the point of a watermark is to not transfer
    the rows again — filtering after the read would move the whole table across the wire every day
    to throw most of it away.
    """
    projection = ", ".join(table.columns)
    sql = f"SELECT {projection} FROM {schema}.{table.source}"
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
    connection: LakebaseConnection,
    table: HistoryTable,
    since: datetime | None,
    token: str,
) -> list[dict]:
    """Read one Postgres table over JDBC and return plain dicts.

    Collected to the driver deliberately: this is a watchlist and a report log, tens of rows, and
    plain dicts are what :func:`merge_rows` takes and what a test can fake without Spark.
    """
    options = jdbc_options(connection, token)
    frame = (
        spark.read.format("jdbc")
        .options(**options)
        .option("query", source_query(table, connection.schema, since))
        .load()
    )
    return [{column: row[column] for column in table.columns} for row in frame.collect()]


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
    connection: LakebaseConnection,
    table: HistoryTable,
    *,
    token: str,
    captured_at: datetime,
    read: Callable[..., Sequence[Mapping]] = read_source,
) -> dict:
    """Sync one table: read the watermark, fetch what is newer, MERGE it, report what moved."""
    target_fqn = qualified(catalog, table.target)
    require_table(spark, target_fqn)

    since = watermark(spark, catalog, table)
    source = read(spark, connection, table, since, token)
    rows = history_rows(source, table, captured_at)

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
    workspace_client: Any = None,
    connection: LakebaseConnection | None = None,
    read: Callable[..., Sequence[Mapping]] = read_source,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Capture both Lakebase tables into Delta (spec C-6).

    One token for the run, one ``captured_at`` for the run, one ``bronze.ingestion_runs`` row for
    the run. A shared ``captured_at`` matters: it is what makes "these rows arrived in the same
    sync" a readable fact rather than a millisecond coincidence.
    """
    catalog = catalog or str(config["catalog"])
    connection = connection or connection_from_config(config, env)

    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    captured_at = utc_now()
    results: list[dict] = []

    try:
        token = access_token(connection, workspace_client)
        for table in HISTORY_TABLES:
            result = sync_table(
                spark,
                catalog,
                connection,
                table,
                token=token,
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
    """A Postgres TIMESTAMPTZ literal from a ``datetime``, and from nothing else.

    The type check is the security control, the same as in ``news_recent``: the JDBC ``query``
    option is text with no parameter markers, so this is what keeps a value from becoming SQL.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"expected a datetime, got {type(value).__name__}: {value!r}")
    moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return f"TIMESTAMP '{moment.astimezone(timezone.utc).isoformat()}'"


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
