"""Lakebase access-layer tests (spec C-2). A fake connection — no Postgres, no workspace.

What these assert is the part of C-2 that is a decision rather than a query result:

- PARAMETERIZED SQL ONLY. Every value travels in the parameter tuple, and never appears in the
  statement text. Ticker names arrive from an LLM tool call, i.e. from user text.
- FULLY QUALIFIED NAMES ONLY. Every statement says ``market_system.<table>``. ``search_path`` is
  not a contract.
- THE PROVEN POOL PATTERN. ``max_lifetime`` under the credential's lifetime, a checkout check,
  and no static password in the connection kwargs.

The live database is exercised by the opt-in test at the bottom and by C-7's integration test.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import pytest
from psycopg_pool import ConnectionPool

from src.database import lakebase
from src.database.lakebase import (
    DEMO_USER_ID,
    DEMO_WATCHLIST_ID,
    POOL_MAX_LIFETIME_SECONDS,
    REPORTS_TABLE,
    SCHEMA,
    USERS_TABLE,
    WATCHLIST_TICKERS_TABLE,
    WATCHLISTS_TABLE,
    LakebaseConfigError,
    LakebaseSettings,
    add_ticker,
    create_lakebase_sql,
    databricks_credential_provider,
    ensure_tables,
    get_watchlist,
    pool_kwargs,
    remove_ticker,
    save_report,
    seed_demo,
    settings_from_env,
)

SQL_PATH = Path(__file__).resolve().parents[1] / "setup" / "create_lakebase.sql"
TABLE_NAMES = ("users", "watchlists", "watchlist_tickers", "research_reports")

SETTINGS = LakebaseSettings(
    host="ep-lingering-brook-d8izukis.database.us-east-2.cloud.databricks.com",
    database="databricks_postgres",
    user="demo@example.com",
    endpoint="projects/regime-market-database/branches/production/endpoints/primary",
)


# --------------------------------------------------------------------------- fakes


@dataclass
class Executed:
    sql: str
    params: tuple | None


class FakeCursor:
    def __init__(self, connection: "FakeConnection"):
        self._connection = connection

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self._connection.executed.append(Executed(sql=sql, params=params))

    def fetchall(self) -> list[tuple]:
        return list(self._connection.rows)


@dataclass
class FakeConnection:
    """Records every statement and its parameters. Reads return ``rows``."""

    rows: list[tuple] = field(default_factory=list)
    executed: list[Executed] = field(default_factory=list)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def statements(self) -> list[str]:
        return [item.sql for item in self.executed]


def only(conn: FakeConnection, keyword: str) -> Executed:
    """The single statement starting with ``keyword``."""
    matches = [item for item in conn.executed if item.sql.lstrip().upper().startswith(keyword)]
    assert len(matches) == 1, f"expected one {keyword} statement, got {len(matches)}"
    return matches[0]


def assert_fully_qualified(conn: FakeConnection) -> None:
    """No statement may name a table without its schema."""
    for statement in conn.statements():
        for table in TABLE_NAMES:
            for match in re.finditer(rf"(?<![\w.]){table}\b", statement):
                prefix = statement[max(0, match.start() - len(SCHEMA) - 1) : match.start()]
                assert prefix == f"{SCHEMA}.", (
                    f"unqualified reference to {table!r} in: {statement}"
                )


def assert_no_values_inline(conn: FakeConnection, *values: str) -> None:
    """A value must reach Postgres through the parameter tuple, never through the SQL text.

    The DDL script is exempt: it is the setup file executed verbatim, and its comments legitimately
    mention tickers.
    """
    for item in conn.executed:
        if item.params is None:
            continue
        for value in values:
            assert value not in item.sql, f"{value!r} was interpolated into: {item.sql}"


# ------------------------------------------------------------------------ watchlist


def test_add_ticker_inserts_parameterized_and_returns_the_new_list():
    conn = FakeConnection(rows=[("MSFT",), ("NVDA",)])

    watchlist = add_ticker("NVDA", conn=conn)

    insert = only(conn, "INSERT")
    assert f"INSERT INTO {WATCHLIST_TICKERS_TABLE}" in insert.sql
    assert insert.params == (DEMO_WATCHLIST_ID, "NVDA", DEMO_USER_ID)
    assert "%s" in insert.sql
    assert_no_values_inline(conn, "NVDA", DEMO_WATCHLIST_ID)
    assert_fully_qualified(conn)
    assert watchlist == ["MSFT", "NVDA"]


def test_adding_the_same_ticker_twice_is_a_no_op_in_sql():
    conn = FakeConnection()

    add_ticker("AMD", conn=conn)

    # Idempotence lives in the statement, not in a read-then-write race in Python.
    assert "ON CONFLICT (watchlist_id, ticker) DO NOTHING" in only(conn, "INSERT").sql


@pytest.mark.parametrize("given,expected", [("nvda", "NVDA"), (" amd ", "AMD"), ("brk.b", "BRK.B")])
def test_tickers_are_normalized(given, expected):
    conn = FakeConnection()

    add_ticker(given, conn=conn)

    assert only(conn, "INSERT").params[1] == expected


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "add tesla", "NVDA'; DROP TABLE market_system.users; --", "TOOLONGTICKER", "1NVDA"],
)
def test_junk_tickers_are_rejected_before_any_statement_runs(bad):
    conn = FakeConnection()

    with pytest.raises(ValueError):
        add_ticker(bad, conn=conn)

    assert conn.executed == []


def test_remove_ticker_deletes_parameterized():
    conn = FakeConnection(rows=[("MSFT",)])

    watchlist = remove_ticker("NVDA", conn=conn)

    delete = only(conn, "DELETE")
    assert f"DELETE FROM {WATCHLIST_TICKERS_TABLE}" in delete.sql
    assert delete.params == (DEMO_WATCHLIST_ID, "NVDA")
    assert_no_values_inline(conn, "NVDA")
    assert watchlist == ["MSFT"]


def test_get_watchlist_reads_one_watchlist_in_ticker_order():
    conn = FakeConnection(rows=[("AMZN",), ("GOOGL",)])

    watchlist = get_watchlist("other-watchlist", conn=conn)

    select = only(conn, "SELECT")
    assert f"FROM {WATCHLIST_TICKERS_TABLE}" in select.sql
    assert "ORDER BY ticker" in select.sql
    assert select.params == ("other-watchlist",)
    assert_no_values_inline(conn, "other-watchlist")
    assert watchlist == ["AMZN", "GOOGL"]


def test_an_empty_watchlist_reads_as_an_empty_list():
    assert get_watchlist(conn=FakeConnection()) == []


# -------------------------------------------------------------------------- reports


def test_save_report_inserts_every_column_as_a_parameter_and_returns_the_id():
    conn = FakeConnection()
    markdown = "## NVDA\nRisk is elevated; the 80% interval is wide -- and it's quoted."

    report_id = save_report(
        "nvda",
        "Why is downside risk elevated?",
        markdown,
        forecast_id="0f9c1a2b-3c4d-5e6f-8091-a2b3c4d5e6f7",
        conn=conn,
    )

    insert = only(conn, "INSERT")
    assert f"INSERT INTO {REPORTS_TABLE}" in insert.sql
    assert insert.params == (
        report_id,
        DEMO_USER_ID,
        "NVDA",
        "Why is downside risk elevated?",
        markdown,
        "0f9c1a2b-3c4d-5e6f-8091-a2b3c4d5e6f7",
    )
    # Report bodies are model-written markdown full of quotes and dashes. They belong in params.
    assert_no_values_inline(conn, markdown, "Why is downside risk elevated?")
    assert_fully_qualified(conn)
    assert len(report_id) == 36


def test_an_explicit_report_id_is_honoured():
    conn = FakeConnection()

    assert save_report("NVDA", "q", "body", report_id="report-1", conn=conn) == "report-1"


def test_an_empty_report_is_refused():
    conn = FakeConnection()

    with pytest.raises(ValueError, match="report_md"):
        save_report("NVDA", "q", "   ", conn=conn)

    assert conn.executed == []


# ----------------------------------------------------------------- schema and seed


def test_ensure_tables_executes_the_setup_sql_file():
    conn = FakeConnection()

    ensure_tables(conn=conn)

    (executed,) = conn.executed
    assert executed.params is None  # a DDL script, not a parameterized query
    assert executed.sql == SQL_PATH.read_text(encoding="utf-8")


def test_the_setup_sql_creates_the_schema_and_the_four_tables():
    sql = create_lakebase_sql()

    assert f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};" in sql
    for table in (USERS_TABLE, WATCHLISTS_TABLE, WATCHLIST_TICKERS_TABLE, REPORTS_TABLE):
        assert f"CREATE TABLE IF NOT EXISTS {table} (" in sql
    assert "PRIMARY KEY (watchlist_id, ticker)" in sql


def test_the_setup_sql_sets_replica_identity_full_on_both_cdc_tables():
    sql = create_lakebase_sql()

    # Lakebase CDF requirement: without FULL, an UPDATE or DELETE reaches the Delta history table
    # carrying only the primary key, and the CDC demo shows a row of nulls.
    for table in (WATCHLIST_TICKERS_TABLE, REPORTS_TABLE):
        assert re.search(rf"ALTER TABLE {re.escape(table)}\s+REPLICA IDENTITY FULL;", sql)


def test_seed_demo_writes_one_user_one_watchlist_and_its_tickers():
    conn = FakeConnection(rows=[("MSFT",), ("NVDA",)])

    watchlist = seed_demo(["NVDA", "MSFT"], conn=conn)

    inserts = [item for item in conn.executed if item.sql.startswith("INSERT")]
    assert f"INSERT INTO {USERS_TABLE}" in inserts[0].sql
    assert inserts[0].params == (DEMO_USER_ID, lakebase.DEMO_USER_NAME)
    assert f"INSERT INTO {WATCHLISTS_TABLE}" in inserts[1].sql
    assert inserts[1].params == (DEMO_WATCHLIST_ID, DEMO_USER_ID, lakebase.DEMO_WATCHLIST_NAME)
    assert [item.params[1] for item in inserts[2:]] == ["NVDA", "MSFT"]
    # Re-running before a demo must not duplicate anything or clobber a hand-added ticker.
    assert all("ON CONFLICT" in item.sql for item in inserts)
    assert_fully_qualified(conn)
    assert watchlist == ["MSFT", "NVDA"]


def test_seeding_no_tickers_writes_only_the_user_and_the_watchlist():
    conn = FakeConnection()

    seed_demo(conn=conn)

    inserts = [item for item in conn.executed if item.sql.startswith("INSERT")]
    assert [USERS_TABLE in item.sql for item in inserts] == [True, False]
    assert [WATCHLISTS_TABLE in item.sql for item in inserts] == [False, True]


# ----------------------------------------------------------------------- settings


def test_settings_come_from_the_local_pg_variables():
    settings = settings_from_env(
        {
            "PGHOST": "host.example",
            "PGPORT": "5433",
            "PGDATABASE": "databricks_postgres",
            "PGUSER": "someone@example.com",
            "PGSSLMODE": "require",
            "LAKEBASE_ENDPOINT": SETTINGS.endpoint,
        }
    )

    assert (settings.host, settings.port, settings.user) == (
        "host.example",
        5433,
        "someone@example.com",
    )
    assert settings.endpoint == SETTINGS.endpoint


def test_settings_fall_back_to_the_app_variables():
    settings = settings_from_env(
        {
            "LAKEBASE_HOST": "app-host.example",
            "LAKEBASE_DATABASE": "databricks_postgres",
            "LAKEBASE_USER": "app-service-principal",
            "LAKEBASE_ENDPOINT": SETTINGS.endpoint,
        }
    )

    assert settings.host == "app-host.example"
    assert settings.port == 5432


def test_incomplete_settings_name_what_is_missing():
    with pytest.raises(LakebaseConfigError) as excinfo:
        settings_from_env({"PGHOST": "host.example"})

    message = str(excinfo.value)
    assert "PGUSER" in message
    assert "LAKEBASE_ENDPOINT" in message
    assert "password" in message  # points out that there is deliberately no password variable


# --------------------------------------------------------------- pool and credential


def test_the_pool_follows_the_proven_pattern():
    kwargs = pool_kwargs(SETTINGS, lambda: "token")

    assert kwargs["max_lifetime"] == POOL_MAX_LIFETIME_SECONDS == 3000
    assert kwargs["check"] is ConnectionPool.check_connection
    assert issubclass(kwargs["connection_class"], psycopg.Connection)
    connection_kwargs = kwargs["kwargs"]
    assert connection_kwargs["sslmode"] == "require"
    assert connection_kwargs["dbname"] == SETTINGS.database
    # The whole point: no static password anywhere in the pool's configuration.
    assert "password" not in connection_kwargs


def test_every_connection_mints_its_own_credential(monkeypatch):
    minted: list[str] = []
    captured: list[dict] = []

    def credential() -> str:
        minted.append(f"token-{len(minted) + 1}")
        return minted[-1]

    @classmethod
    def fake_connect(_cls, conninfo="", **kwargs):
        captured.append(kwargs)
        return "connection"

    monkeypatch.setattr(psycopg.Connection, "connect", fake_connect)
    connection_class = pool_kwargs(SETTINGS, credential)["connection_class"]

    connection_class.connect("", **SETTINGS.connection_kwargs())
    connection_class.connect("", **SETTINGS.connection_kwargs())

    # A token reused across the pool's lifetime is the failure mode max_lifetime exists for.
    assert [item["password"] for item in captured] == ["token-1", "token-2"]


def test_the_credential_provider_asks_for_the_configured_endpoint():
    asked: list[str] = []

    class FakePostgres:
        def generate_database_credential(self, *, endpoint):
            asked.append(endpoint)
            return type("Credential", (), {"token": "oauth-token"})()

    class FakeWorkspace:
        postgres = FakePostgres()

    provider = databricks_credential_provider(SETTINGS.endpoint, workspace_client=FakeWorkspace())

    assert provider() == "oauth-token"
    assert asked == [SETTINGS.endpoint]


def test_a_credential_without_a_token_fails_loudly():
    class FakeWorkspace:
        class postgres:  # noqa: N801 — mirrors the SDK attribute name
            @staticmethod
            def generate_database_credential(*, endpoint):
                return type("Credential", (), {"token": None})()

    provider = databricks_credential_provider(SETTINGS.endpoint, workspace_client=FakeWorkspace())

    with pytest.raises(LakebaseConfigError, match="no token"):
        provider()


# --------------------------------------------------------------------- integration
# The whole demo write sequence over one connection, through the real functions: the C-2 setup,
# the seed, the CDC-demo add, the read the sidebar makes, a remove, and a saved report.


def test_the_demo_write_sequence_runs_end_to_end_over_one_connection():
    conn = FakeConnection(rows=[("AMD",), ("NVDA",)])

    ensure_tables(conn=conn)
    seed_demo(["NVDA"], conn=conn)
    after_add = add_ticker("AMD", conn=conn)
    assert get_watchlist(conn=conn) == after_add
    remove_ticker("AMD", conn=conn)
    report_id = save_report("NVDA", "Why is risk elevated?", "## NVDA\nBecause...", conn=conn)

    assert report_id
    assert_fully_qualified(conn)
    assert_no_values_inline(conn, "AMD", "NVDA", "Why is risk elevated?")
    # Reads and writes alike are parameterized: only the DDL script runs without parameters.
    assert [item.params for item in conn.executed].count(None) == 1


# ------------------------------------------------------------------- live (opt-in)
# Needs the real Lakebase project, a Postgres role for the caller, and grants on market_system.
# This is the C-d verify-first item "Lakebase project creatable", kept runnable:
#   LAKEBASE_LIVE_TEST=1 .venv/Scripts/python.exe -m pytest tests/test_lakebase.py -k live -q


@pytest.mark.skipif(os.environ.get("LAKEBASE_LIVE_TEST") != "1", reason="set LAKEBASE_LIVE_TEST=1")
def test_live_lakebase_round_trip():
    ensure_tables()
    seed_demo(["NVDA"])

    try:
        assert "AMD" in add_ticker("AMD")
        assert "AMD" in get_watchlist()
        assert "AMD" in add_ticker("AMD")  # idempotent
        assert "AMD" not in remove_ticker("AMD")

        report_id = save_report("NVDA", "live test", "## live test\nbody")
        assert report_id
    finally:
        lakebase.close_pool()
