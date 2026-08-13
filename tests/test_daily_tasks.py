"""The three daily tasks added at C-6: ``fit_models``, ``news_recent`` and ``lakebase_history``.

All three are Spark-facing, and all three are tested against the shared ``FakeSpark`` rather than
a cluster. That fake never parses SQL, so what these tests can prove is bounded and stated:

- WHAT IS PROVEN: the control flow (one ledger row per task per run, on success and on failure),
  the fact that every write is a MERGE, the exact window arithmetic, the watermark comparison, and
  the row shapes handed to the write layer. Those are Python, not SQL.
- WHAT IS NOT: that the SQL is valid Databricks SQL. Only a workspace can say that.

The window arithmetic gets the most attention because it is where the two ``news_recent`` tasks
can quietly undo each other — a daily refresh deleting exactly what the backfill added is a bug
that costs a warehouse every night and shows up as "the page's news list never gets longer".
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.pipelines import GOLD_FORECAST_RUNS, GOLD_REGIME_STATES, fit_models, lakebase_history
from src.pipelines import news_recent as nr
from tests.conftest import FakeResult, FakeSpark, backtest_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL = (REPO_ROOT / "setup" / "create_delta_tables.sql").read_text(encoding="utf-8")

TODAY = date(2026, 8, 12)
NOW = datetime(2026, 8, 12, 22, 30, tzinfo=timezone.utc)


def config(**overrides) -> dict:
    """A config.yaml-shaped mapping with the C-6 blocks the shipped file carries."""
    resolved = backtest_config()
    resolved["news_recent"] = {
        "window_days": 90,
        "batch_days": 90,
        "backfill_floor": "2024-08-01",
        "include_backfill": False,
    }
    resolved["lakebase"] = {
        "host": "lakebase.example.databricks.com",
        "port": 5432,
        "database": "databricks_postgres",
        "user": "app-sp",
        "schema": "market_system",
        "endpoint": "projects/regime-market-database/branches/production/endpoints/primary",
    }
    for key, value in overrides.items():
        resolved[key] = {**resolved.get(key, {}), **value} if isinstance(value, dict) else value
    return resolved


class AnsweringSpark(FakeSpark):
    """``FakeSpark`` plus canned answers, matched on a substring of the statement."""

    def __init__(self, answers: dict[str, object] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.answers = dict(answers or {})

    def sql(self, text: str, args: dict | None = None):
        self.statements.append(text)
        if self._fail_on and self._fail_on in text:
            raise RuntimeError("boom: simulated Spark failure")
        for needle, answer in self.answers.items():
            if needle in text:
                return answer
        if "count(*) AS n FROM" in text:
            return FakeResult({"n": self._row_count})
        return FakeResult()

    def deletes(self) -> list[str]:
        return [text for text in self.statements if text.lstrip().startswith("DELETE FROM")]

    def data_merges(self) -> list[str]:
        """MERGEs into the tables under test, minus the ledger — which is itself a MERGE."""
        return [text for text in self.merge_statements() if "bronze.ingestion_runs" not in text]


# ===================================================================== news_recent


class TestNewsRecentWindow:
    """The window arithmetic, which is the whole design decision, as pure functions."""

    def test_the_rolling_window_is_the_configured_number_of_days(self):
        settings = nr.settings_from_config(config())
        assert nr.window_start(TODAY, settings) == date(2026, 5, 14)

    def test_without_the_backfill_the_floor_is_the_rolling_window(self):
        settings = nr.settings_from_config(config())
        assert nr.retention_floor(TODAY, settings) == nr.window_start(TODAY, settings)

    def test_with_the_backfill_on_the_floor_drops_to_the_backfill_floor(self):
        """Otherwise the daily refresh deletes exactly what the backfill spent runs adding."""
        settings = nr.settings_from_config(config(news_recent={"include_backfill": True}))
        assert nr.retention_floor(TODAY, settings) == date(2024, 8, 1)

    def test_the_floor_is_never_above_the_rolling_window(self):
        """A backfill_floor in the future must not shorten the window the app depends on."""
        settings = nr.settings_from_config(
            config(news_recent={"include_backfill": True, "backfill_floor": "2026-08-11"})
        )
        assert nr.retention_floor(TODAY, settings) == date(2026, 5, 14)

    def test_the_first_backfill_batch_starts_at_the_rolling_edge(self):
        settings = nr.settings_from_config(config())
        assert nr.next_backfill_window(None, TODAY, settings) == (date(2026, 2, 13), date(2026, 5, 14))

    def test_each_run_moves_one_batch_further_back(self):
        settings = nr.settings_from_config(config())
        first = nr.next_backfill_window(date(2026, 5, 14), TODAY, settings)
        second = nr.next_backfill_window(first[0], TODAY, settings)

        assert first == (date(2026, 2, 13), date(2026, 5, 14))
        assert second == (date(2025, 11, 15), date(2026, 2, 13))
        assert first[0] == second[1], "the batches must abut, with no day read twice"

    def test_the_last_batch_stops_exactly_on_the_floor(self):
        settings = nr.settings_from_config(config())
        assert nr.next_backfill_window(date(2024, 9, 1), TODAY, settings) == (
            date(2024, 8, 1),
            date(2024, 9, 1),
        )

    def test_at_the_floor_there_is_nothing_left_to_do(self):
        settings = nr.settings_from_config(config())
        assert nr.next_backfill_window(date(2024, 8, 1), TODAY, settings) is None
        assert nr.next_backfill_window(date(2024, 7, 1), TODAY, settings) is None


class TestNewsRecentSql:
    def test_the_projection_is_exactly_the_target_columns(self):
        """``UPDATE SET * / INSERT *`` matches by name; a missing column fails the write."""
        block = DDL.split("CREATE TABLE IF NOT EXISTS market_intel.silver.news_recent (", 1)[1]
        block = block.split(")\nUSING DELTA", 1)[0]
        declared = tuple(
            line.strip().split()[0] for line in block.splitlines() if line.strip() and not line.strip().startswith("--")
        )
        assert nr.COLUMNS == declared

    def test_the_refresh_query_has_a_lower_bound_and_no_upper_one(self):
        sql = nr.source_sql("market_intel", date(2026, 5, 14))

        assert "a.published_at >= DATE '2026-05-14'" in sql
        assert "published_at < DATE" not in sql
        assert "FROM market_intel.silver.news_articles" in sql

    def test_rows_the_window_already_holds_unchanged_are_not_re_presented(self):
        """Otherwise every night's MERGE rewrites the whole window.

        A rewritten row is a Change Data Feed event, the AI Search index (C-1) is built on this
        table, and a feed event is an embedding call — so the naive refresh re-embeds thousands of
        unchanged articles daily.
        """
        sql = nr.source_sql("market_intel", date(2026, 5, 14))

        assert "NOT EXISTS (SELECT 1 FROM market_intel.silver.news_recent AS held" in sql
        assert "held.`article_id` = a.`article_id`" in sql
        assert "held.`ticker` = a.`ticker`" in sql

    def test_the_anti_join_does_not_shadow_the_merge_target_alias(self):
        # merge_sql wraps this text in `MERGE INTO ... AS t USING (<source>) AS s`.
        assert " AS t\n" not in nr.source_sql("market_intel", date(2026, 5, 14))

    def test_a_row_that_did_change_still_gets_through(self):
        """Every non-key column is compared, not just the embedding source.

        Skipping only on the embedding text would freeze a corrected sentiment label in the app's
        news list forever, since nothing else ever rewrites this table.
        """
        predicate = nr.unchanged_predicate("market_intel")

        for column in nr.COLUMNS:
            if column in nr.MERGE_KEYS:
                continue
            assert f"held.`{column}` <=> a.`{column}`" in predicate, f"{column} is not compared"

    def test_the_comparison_is_null_safe(self):
        """`=` on two NULLs is NULL, not true.

        With a plain `=`, an article whose publisher is NULL — the vendor sends those — would fail
        its own comparison and be re-presented every night forever, which is exactly the churn the
        predicate exists to stop.
        """
        predicate = nr.unchanged_predicate("market_intel")
        # One comparison per line, except the key join, which is the WHERE line.
        comparisons = [
            line for line in predicate.splitlines() if line.strip().startswith("AND held.")
        ]

        assert comparisons, "no column comparisons at all"
        assert all("<=>" in line for line in comparisons)

    def test_a_backfill_batch_is_bounded_at_both_ends(self):
        sql = nr.source_sql("market_intel", date(2026, 2, 13), date(2026, 5, 14))

        assert "published_at >= DATE '2026-02-13'" in sql
        assert "published_at < DATE '2026-05-14'" in sql

    def test_only_a_real_date_can_reach_the_sql(self):
        """The MERGE source has no parameter markers, so the type check is the control."""
        with pytest.raises(TypeError):
            nr.source_sql("market_intel", "2026-05-14'; DROP TABLE x --")
        with pytest.raises(TypeError):
            nr.source_sql("market_intel", datetime(2026, 5, 14))


class TestNewsRecentTasks:
    def test_the_refresh_merges_then_deletes(self, monkeypatch):
        """In that order: the other way leaves the page with a hole for the length of the merge."""
        monkeypatch.setattr(nr, "utc_now", lambda: NOW)
        spark = AnsweringSpark()

        result = nr.refresh(spark, config())

        merge_index = next(i for i, text in enumerate(spark.statements) if text.startswith("MERGE INTO"))
        delete_index = next(i for i, text in enumerate(spark.statements) if text.startswith("DELETE FROM"))
        assert merge_index < delete_index
        assert "MERGE INTO market_intel.silver.news_recent" in spark.statements[merge_index]
        assert "WHERE published_at < DATE '2026-05-14'" in spark.statements[delete_index]
        assert result["rows_merged"] == 7

    def test_the_refresh_merges_on_the_declared_keys(self):
        spark = AnsweringSpark()

        nr.refresh(spark, config())

        merge = spark.data_merges()[0]
        assert "t.`article_id` = s.`article_id`" in merge
        assert "t.`ticker` = s.`ticker`" in merge
        assert "INSERT INTO" not in merge

    def test_the_refresh_leaves_one_ledger_row(self, monkeypatch):
        monkeypatch.setattr(nr, "utc_now", lambda: NOW)
        spark = AnsweringSpark()

        nr.refresh(spark, config())

        rows = spark.ledger_rows()
        assert len(rows) == 1
        assert rows[0]["task"] == "refresh_news_recent"
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["rows_written"] == 7

    def test_a_failed_refresh_still_leaves_a_ledger_row(self):
        spark = AnsweringSpark(fail_on="MERGE INTO")

        with pytest.raises(RuntimeError):
            nr.refresh(spark, config())

        rows = spark.ledger_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "boom" in rows[0]["error"]

    def test_the_backfill_reads_its_cursor_from_the_table(self):
        spark = AnsweringSpark({"MIN(published_at)": FakeResult({"oldest": date(2026, 5, 14)})})

        result = nr.backfill(spark, config())

        assert result["done"] is False
        assert result["window"] == {"start": date(2026, 2, 13), "end": date(2026, 5, 14)}
        merge = spark.data_merges()[0]
        assert "published_at >= DATE '2026-02-13'" in merge
        assert "published_at < DATE '2026-05-14'" in merge

    def test_a_finished_backfill_is_a_no_op_that_still_reports(self):
        """It must not start failing the day it finishes — a scheduled task lives past its work."""
        spark = AnsweringSpark({"MIN(published_at)": FakeResult({"oldest": date(2024, 8, 1)})})

        result = nr.backfill(spark, config())

        assert result["done"] is True
        assert result["rows_merged"] == 0
        assert spark.data_merges() == []
        assert spark.ledger_rows()[0]["task"] == "backfill_news_recent"

    def test_the_backfill_never_deletes(self):
        spark = AnsweringSpark({"MIN(published_at)": FakeResult({"oldest": date(2026, 5, 14)})})

        nr.backfill(spark, config())

        assert spark.deletes() == []


# ===================================================================== lakebase_history


class FakeCredential:
    def __init__(self, token: str = "oauth-token-abc"):
        self.token = token


class FakePostgresAPI:
    def __init__(self):
        self.calls: list[str] = []

    def generate_database_credential(self, endpoint: str):
        self.calls.append(endpoint)
        return FakeCredential()


class FakeWorkspaceClient:
    def __init__(self):
        self.postgres = FakePostgresAPI()


class FakeReader:
    """Stands in for the JDBC read. Records what it was asked for; returns canned Postgres rows."""

    def __init__(self, rows: dict[str, list[dict]] | None = None):
        self.rows = dict(rows or {})
        self.calls: list[dict] = []

    def __call__(self, spark, connection, table, since, token):
        self.calls.append(
            {"table": table.source, "since": since, "token": token, "connection": connection}
        )
        return self.rows.get(table.source, [])


WATCHLIST_ROW = {
    "watchlist_id": "demo-watchlist",
    "ticker": "AMD",
    "added_at": datetime(2026, 8, 12, 20, 5, tzinfo=timezone.utc),
    "added_by": "demo-user",
}
REPORT_ROW = {
    "report_id": "report-1",
    "user_id": "demo-user",
    "ticker": "NVDA",
    "question": "why is downside risk elevated?",
    "report_md": "# NVDA\nBecause...",
    "forecast_id": "forecast-123",
    "created_at": datetime(2026, 8, 12, 20, 6, tzinfo=timezone.utc),
}


class TestLakebaseConnection:
    def test_the_url_requires_tls(self):
        connection = lakebase_history.connection_from_config(config(), env={})
        assert connection.jdbc_url() == (
            "jdbc:postgresql://lakebase.example.databricks.com:5432/databricks_postgres?sslmode=require"
        )

    def test_the_environment_wins_over_the_checked_in_config(self):
        connection = lakebase_history.connection_from_config(
            config(), env={"PGHOST": "override.example.com", "PGUSER": "someone-else"}
        )
        assert connection.host == "override.example.com"
        assert connection.user == "someone-else"

    def test_missing_settings_fail_with_the_names_to_fix(self):
        with pytest.raises(ValueError, match="host, user"):
            lakebase_history.connection_from_config(
                config(lakebase={"host": "", "user": ""}), env={}
            )

    def test_the_password_is_minted_and_never_stored(self):
        """The whole reason this task can exist without a secret: a per-run OAuth token."""
        connection = lakebase_history.connection_from_config(config(), env={})
        client = FakeWorkspaceClient()

        token = lakebase_history.access_token(connection, client)

        assert token == "oauth-token-abc"
        assert client.postgres.calls == [connection.endpoint]
        options = lakebase_history.jdbc_options(connection, token)
        assert options["password"] == token
        assert token not in options["url"], "the token must not travel in the URL"

    def test_a_credential_without_a_token_is_an_error_not_a_none_password(self):
        connection = lakebase_history.connection_from_config(config(), env={})
        client = FakeWorkspaceClient()
        client.postgres.generate_database_credential = lambda endpoint: FakeCredential("")

        with pytest.raises(RuntimeError, match="no token"):
            lakebase_history.access_token(connection, client)


class TestWatermarkSync:
    def test_the_first_run_reads_the_whole_table(self):
        spark = AnsweringSpark()  # max(...) returns no row at all
        reader = FakeReader()

        lakebase_history.main(spark, config(), workspace_client=FakeWorkspaceClient(), read=reader)

        assert [call["since"] for call in reader.calls] == [None, None]
        for table in lakebase_history.HISTORY_TABLES:
            assert lakebase_history.source_query(table, "market_system", None).count("WHERE") == 0

    def test_a_later_run_reads_only_what_is_newer(self):
        watermark = datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc)
        spark = AnsweringSpark({"max(added_at)": FakeResult({"watermark": watermark})})
        reader = FakeReader()

        lakebase_history.main(spark, config(), workspace_client=FakeWorkspaceClient(), read=reader)

        watchlist_call = next(call for call in reader.calls if call["table"] == "watchlist_tickers")
        assert watchlist_call["since"] == watermark

    def test_the_watermark_is_pushed_down_to_postgres(self):
        table = lakebase_history.HISTORY_TABLES[0]
        since = datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc)

        sql = lakebase_history.source_query(table, "market_system", since)

        assert "FROM market_system.watchlist_tickers" in sql
        assert "WHERE added_at >= TIMESTAMP '2026-08-11T18:00:00+00:00'" in sql

    def test_the_comparison_includes_the_boundary_row(self):
        """``>`` would drop a second row sharing the boundary timestamp, forever."""
        table = lakebase_history.HISTORY_TABLES[0]
        sql = lakebase_history.source_query(table, "market_system", datetime(2026, 8, 11))
        assert ">=" in sql and " > " not in sql

    def test_a_naive_watermark_is_read_as_utc(self):
        table = lakebase_history.HISTORY_TABLES[1]
        sql = lakebase_history.source_query(table, "market_system", datetime(2026, 8, 11, 18, 0))
        assert "'2026-08-11T18:00:00+00:00'" in sql

    def test_only_a_datetime_can_reach_the_query(self):
        table = lakebase_history.HISTORY_TABLES[0]
        with pytest.raises(TypeError):
            lakebase_history.source_query(table, "market_system", "2026-08-11' OR 1=1 --")

    def test_captured_rows_carry_the_run_timestamp(self, monkeypatch):
        monkeypatch.setattr(lakebase_history, "utc_now", lambda: NOW)
        spark = AnsweringSpark()
        reader = FakeReader(
            {"watchlist_tickers": [WATCHLIST_ROW], "research_reports": [REPORT_ROW]}
        )

        result = lakebase_history.main(
            spark, config(), workspace_client=FakeWorkspaceClient(), read=reader
        )

        assert result["captured_at"] == NOW
        staged = [frame for frame in spark.frames if "captured_at" in frame.schema]
        assert len(staged) == 2
        for frame in staged:
            assert all(row[-1] == NOW for row in frame.rows), "captured_at is the last column"

    def test_both_tables_share_one_token_and_one_capture_time(self):
        """A shared stamp is what makes "these arrived together" a readable fact."""
        client = FakeWorkspaceClient()
        reader = FakeReader({"watchlist_tickers": [WATCHLIST_ROW]})

        lakebase_history.main(AnsweringSpark(), config(), workspace_client=client, read=reader)

        assert len(client.postgres.calls) == 1
        assert {call["token"] for call in reader.calls} == {"oauth-token-abc"}

    def test_the_write_is_a_merge_on_the_source_primary_key(self):
        spark = AnsweringSpark()
        reader = FakeReader({"watchlist_tickers": [WATCHLIST_ROW], "research_reports": [REPORT_ROW]})

        lakebase_history.main(spark, config(), workspace_client=FakeWorkspaceClient(), read=reader)

        merges = spark.data_merges()
        assert len(merges) == 2
        assert "t.watchlist_id = s.watchlist_id AND t.ticker = s.ticker" in merges[0]
        assert "t.report_id = s.report_id" in merges[1]

    def test_it_writes_into_the_gold_history_tables(self):
        spark = AnsweringSpark()
        reader = FakeReader({"watchlist_tickers": [WATCHLIST_ROW], "research_reports": [REPORT_ROW]})

        lakebase_history.main(spark, config(), workspace_client=FakeWorkspaceClient(), read=reader)

        merged = " ".join(spark.data_merges())
        assert "market_intel.gold.lb_watchlist_tickers_history" in merged
        assert "market_intel.gold.lb_research_reports_history" in merged

    def test_nothing_new_writes_nothing_but_still_ledgers(self):
        spark = AnsweringSpark()

        result = lakebase_history.main(
            spark, config(), workspace_client=FakeWorkspaceClient(), read=FakeReader()
        )

        assert result["rows_total"] == 0
        assert spark.data_merges() == []
        assert spark.ledger_rows()[0]["task"] == "sync_lakebase_history"

    def test_a_failure_is_recorded_before_it_propagates(self):
        spark = AnsweringSpark(fail_on="MERGE INTO")
        reader = FakeReader({"watchlist_tickers": [WATCHLIST_ROW]})

        with pytest.raises(RuntimeError):
            lakebase_history.main(
                spark, config(), workspace_client=FakeWorkspaceClient(), read=reader
            )

        assert spark.ledger_rows()[0]["status"] == "failed"

    def test_the_history_schemas_match_the_delta_ddl(self):
        """A column added to the DDL and forgotten here would arrive as a silent NULL."""
        for table in lakebase_history.HISTORY_TABLES:
            fqn = f"market_intel.{table.target}"
            block = DDL.split(f"CREATE TABLE IF NOT EXISTS {fqn} (", 1)[1].split(")\nUSING DELTA", 1)[0]
            declared = tuple(
                line.strip().split()[0]
                for line in block.splitlines()
                if line.strip() and not line.strip().startswith("--")
            )
            assert table.target_columns == declared
            assert [part.split()[0] for part in table.schema_ddl.split(", ")] == list(declared)

    def test_a_delete_in_postgres_removes_nothing_here(self):
        """History, not a mirror: the sync only ever reads forward and merges."""
        spark = AnsweringSpark()
        lakebase_history.main(
            spark, config(), workspace_client=FakeWorkspaceClient(), read=FakeReader()
        )
        assert spark.deletes() == []


# ===================================================================== fit_models


class FeatureSpark(AnsweringSpark):
    """``AnsweringSpark`` that can also hand back a pandas frame at the ``.toPandas()`` boundary."""

    def __init__(self, frames: dict[str, object] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.frames_by_ticker = dict(frames or {})
        self.read_tickers: list[str] = []

    def sql(self, text: str, args: dict | None = None):
        if "FROM market_intel.silver.daily_features" in text and "SELECT DISTINCT" not in text:
            ticker = (args or {})["ticker"]
            self.read_tickers.append(ticker)
            self.statements.append(text)
            return _PandasResult(self.frames_by_ticker.get(ticker))
        if "SELECT DISTINCT ticker" in text:
            self.statements.append(text)
            return FakeResult(rows=[{"ticker": name} for name in sorted(self.frames_by_ticker)])
        return super().sql(text, args)


class _PandasResult:
    def __init__(self, frame):
        self._frame = frame

    def toPandas(self):  # noqa: N802 — Spark's API
        import pandas as pd

        return pd.DataFrame(columns=list(fit_models.FEATURE_COLUMNS)) if self._frame is None else self._frame


class StubArmFit:
    def __init__(self, model_used: str, sorted_params=None, failure_reason=None):
        self.model = "news_markov"
        self.model_used = model_used
        self.converged = True
        self.failure_reason = failure_reason
        self.sorted_params = sorted_params
        self.summary = type(
            "Summary",
            (),
            {
                "model_used": model_used,
                "model_version": "2.1.0",
                "horizon_days": 5,
                "n_paths": 200,
                "seed": 42,
                "current_price": 100.0,
                "price_p10": 95.0,
                "price_p50": 100.5,
                "price_p90": 106.0,
                "return_p10": -0.05,
                "return_p50": 0.005,
                "return_p90": 0.06,
                "prob_positive": 0.52,
                "prob_loss_gt_5pct": 0.09,
                "prob_low_vol": 0.7,
                "prob_high_vol": 0.3,
            },
        )()


class TestFitModels:
    def test_one_fit_produces_a_row_for_each_gold_table(self, backtest_frame, backtest_cfg):
        """The reason the two tables are one task: both rows come from the SAME fitted model."""
        fit = fit_models.fit_ticker("NVDA", backtest_frame, backtest_cfg)

        assert fit.ok
        assert fit.model_used in ("news_markov", "markov", "gbm")
        assert fit.forecast_row["ticker"] == "NVDA"
        assert fit.as_of_date == backtest_frame["trade_date"].iloc[-1]
        if fit.model_used == "gbm":
            assert fit.regime_row is None
        else:
            assert fit.regime_row["as_of_date"] == fit.forecast_row["as_of_date"]
            assert fit.regime_row["model_used"] == fit.forecast_row["model_used"]

    def test_the_fit_is_at_the_last_session_not_the_last_scorable_origin(
        self, backtest_frame, backtest_cfg
    ):
        """Production forecasts from today; only a backtest needs a day whose future has happened."""
        fit = fit_models.fit_ticker("NVDA", backtest_frame, backtest_cfg)
        assert fit.as_of_date == backtest_frame["trade_date"].iloc[-1]

    def test_a_gbm_fallback_writes_a_forecast_but_no_regime_row(self, monkeypatch, backtest_frame):
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: StubArmFit("gbm"))

        fit = fit_models.fit_ticker("NVDA", backtest_frame, config())

        assert fit.forecast_row is not None
        assert fit.regime_row is None

    def test_a_ticker_with_no_features_is_skipped_not_failed(self, backtest_cfg):
        import pandas as pd

        fit = fit_models.fit_ticker("AMD", pd.DataFrame(columns=fit_models.FEATURE_COLUMNS), backtest_cfg)

        assert not fit.ok
        assert "no rows" in fit.skipped_reason

    def test_a_ticker_whose_every_rung_failed_is_reported(self, monkeypatch, backtest_frame):
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: None)

        fit = fit_models.fit_ticker("NVDA", backtest_frame, config())

        assert not fit.ok
        assert "every rung" in fit.skipped_reason

    def test_the_universe_comes_from_the_feature_table(self, monkeypatch, backtest_frame):
        """Not from the watchlist: this task cannot import psycopg (see the module docstring)."""
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: StubArmFit("gbm"))
        spark = FeatureSpark({"AMD": backtest_frame, "NVDA": backtest_frame})

        fit_models.main(spark, config())

        assert spark.read_tickers == ["AMD", "NVDA"]

    def test_an_explicit_ticker_list_overrides_the_table(self, monkeypatch, backtest_frame):
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: StubArmFit("gbm"))
        spark = FeatureSpark({"NVDA": backtest_frame})

        fit_models.main(spark, config(), tickers=["nvda"])

        assert spark.read_tickers == ["NVDA"]

    def test_both_tables_are_merged_in_one_task_with_one_ledger_row(
        self, monkeypatch, backtest_frame, sorted_markov
    ):
        monkeypatch.setattr(
            fit_models, "fit_arm", lambda *a, **k: StubArmFit("news_markov", sorted_markov)
        )
        spark = FeatureSpark({"NVDA": backtest_frame})

        result = fit_models.main(spark, config())

        assert set(result["rows_written"]) == {GOLD_REGIME_STATES, GOLD_FORECAST_RUNS}
        merged = " ".join(spark.data_merges())
        assert "market_intel.gold.regime_states" in merged
        assert "market_intel.gold.forecast_runs" in merged

        rows = spark.ledger_rows()
        assert len(rows) == 1
        assert rows[0]["task"] == "fit_models"

    def test_every_forecast_in_a_run_shares_one_generated_at(
        self, monkeypatch, backtest_frame, sorted_markov
    ):
        monkeypatch.setattr(
            fit_models, "fit_arm", lambda *a, **k: StubArmFit("news_markov", sorted_markov)
        )
        spark = FeatureSpark({"AMD": backtest_frame, "NVDA": backtest_frame})

        fit_models.main(spark, config())

        staged = [frame for frame in spark.frames if "forecast_id STRING" in frame.schema]
        stamps = {row[2] for frame in staged for row in frame.rows}  # generated_at is column 3
        assert len(stamps) == 1

    def test_a_run_where_every_ticker_failed_still_ledgers(self, monkeypatch, backtest_frame):
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: None)
        spark = FeatureSpark({"NVDA": backtest_frame})

        result = fit_models.main(spark, config())

        assert result["skipped"] == [{"ticker": "NVDA", "reason": "every rung of the ladder failed"}]
        assert spark.data_merges() == []
        assert spark.ledger_rows()[0]["task"] == "fit_models"

    def test_the_ticker_travels_as_a_parameter_not_in_the_sql(self, monkeypatch, backtest_frame):
        monkeypatch.setattr(fit_models, "fit_arm", lambda *a, **k: StubArmFit("gbm"))
        spark = FeatureSpark({"NVDA": backtest_frame})

        fit_models.main(spark, config())

        reads = [text for text in spark.statements if "daily_features" in text and "DISTINCT" not in text]
        assert reads and all(":ticker" in text and "'NVDA'" not in text for text in reads)
