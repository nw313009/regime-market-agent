"""Idempotency tests (spec A-5, and rule 6: every pipeline write is idempotent).

- ``test_idempotency``: run the silver build twice over the same bronze data and assert
  identical row counts AND identical checksums. Row counts alone would miss a MERGE that
  updates rows it should have left untouched.
- Extend the same double-run assertion to the bronze ingestion tasks and to the feature
  pipeline.
- Assert every write path is a MERGE on the declared keys, never a blind INSERT:

    bronze.prices_raw       (ticker, source_timestamp)
    bronze.news_raw         (article_id, ticker)
    silver.daily_prices     (ticker, trade_date)
    silver.news_articles    (article_id, ticker)
    silver.daily_features   (ticker, trade_date)
    gold.regime_states      (ticker, as_of_date)
    gold.forecast_runs      (ticker, as_of_date, model_used)
    gold.backtest_metrics   (origin_date, ticker, model)
    gold.backtest_summary   (model)

This matters operationally: the daily job retries, and a retry must not duplicate data.

WHAT IS COVERED HERE NOW, AND WHAT STILL NEEDS A CLUSTER. The gold write layer (B-6) is tested
below against the shared fake SparkSession, because the parts that go wrong are Python: whether the
statement is a MERGE at all, which keys it matches on, whether the row's columns match the DDL, and
whether the ledger gets exactly one row per task. The double-run-and-checksum assertions are
genuinely about Delta's behaviour and are marked TODO(integration) until they can run against the
workspace.

WHY THE COLUMN-SET ASSERTIONS MATTER AS MUCH AS THE MERGE ONES. Rows reach Delta as positional
tuples projected from dicts by column name, so a renamed field does not fail — it lands as a silent
NULL in a published table. The row builders are therefore asserted to produce exactly the DDL's
columns, and the DDL file itself is asserted to declare them.

TODO(integration, workspace): the double-run row-count-and-checksum assertions for bronze, silver
and gold against real Delta tables.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.ingestion import STATUS_FAILED, STATUS_SUCCEEDED
from src.models.backtest import BacktestRow, PooledRow
from src.pipelines import (
    GOLD_BACKTEST_METRICS,
    GOLD_BACKTEST_SUMMARY,
    GOLD_FORECAST_RUNS,
    GOLD_REGIME_STATES,
    GOLD_TABLES,
    forecast_id_for,
    forecast_run_row,
    regime_state_row,
    write_gold,
)
from tests.conftest import FakeSpark

CATALOG = "market_intel"
AS_OF = date(2026, 8, 7)
GENERATED_AT = datetime(2026, 8, 7, 21, 5, tzinfo=timezone.utc)

#: The declared MERGE keys of every gold table. Duplicated from the DDL on purpose: a test that
#: imports the keys it is checking cannot catch a key being changed.
EXPECTED_GOLD_KEYS = {
    GOLD_REGIME_STATES: ("ticker", "as_of_date"),
    GOLD_FORECAST_RUNS: ("ticker", "as_of_date", "model_used"),
    GOLD_BACKTEST_METRICS: ("origin_date", "ticker", "model"),
    GOLD_BACKTEST_SUMMARY: ("model",),
}


def _read_ddl() -> str:
    path = Path(__file__).resolve().parents[1] / "setup" / "create_delta_tables.sql"
    return path.read_text(encoding="utf-8")


def _ddl_block(table: str) -> str:
    """The column declarations of one CREATE TABLE, without the surrounding statement."""
    return _read_ddl().split(f"{CATALOG}.{table} (", 1)[1].split(")\nUSING DELTA", 1)[0]


def metric_row() -> BacktestRow:
    return BacktestRow(
        origin_date=date(2026, 7, 31),
        ticker="NVDA",
        model="news_markov",
        brier=0.21,
        mae=0.013,
        covered_80=True,
        model_used="markov",
        converged=True,
        failure_reason="news_markov: degenerate fit",
        realized_return=0.004,
        return_p50=-0.009,
        prob_positive=0.54,
    )


def pooled_row() -> PooledRow:
    return PooledRow(
        model="news_markov",
        n=130,
        n_tickers=5,
        brier=0.24,
        mae=0.031,
        coverage_80=0.78,
        fallback_rate=0.12,
    )


# ------------------------------------------------------------------ the write contract


@pytest.mark.parametrize("table", sorted(GOLD_TABLES))
def test_every_gold_table_declares_its_merge_keys(table: str):
    """Rule 4: every write is a MERGE on the declared keys, never a blind INSERT."""
    gold = GOLD_TABLES[table]

    assert gold.keys == EXPECTED_GOLD_KEYS[table]
    assert set(gold.keys) <= set(gold.columns)


@pytest.mark.parametrize("table", sorted(GOLD_TABLES))
def test_every_gold_schema_matches_its_ddl_columns(table: str):
    """The write schema and the table it writes into must agree, column for column and in order."""
    gold = GOLD_TABLES[table]
    block = _ddl_block(gold.table)

    declared = re.findall(r"^\s{2}`?(\w+)`?\s+[A-Z]", block, re.MULTILINE)

    assert declared == list(gold.columns)


@pytest.mark.parametrize("table", sorted(GOLD_TABLES))
def test_every_gold_merge_key_is_not_null_in_the_ddl(table: str):
    """A NULL in a MERGE key never matches, which would turn the MERGE into an append."""
    block = _ddl_block(GOLD_TABLES[table].table)

    for key in GOLD_TABLES[table].keys:
        assert re.search(rf"^\s{{2}}`?{key}`?\s+\w+\s+NOT NULL", block, re.MULTILINE), key


@pytest.mark.parametrize("table", sorted(GOLD_TABLES))
def test_no_gold_table_is_partitioned(table: str):
    """These tables are tiny (~2.5k rows per ticker); the defaults are correct."""
    statement = _read_ddl().split(f"{CATALOG}.{GOLD_TABLES[table].table} (", 1)[1]

    assert "PARTITIONED BY" not in statement.split(";", 1)[0]


def test_write_gold_merges_on_the_declared_keys():
    spark = FakeSpark()

    write_gold(
        spark,
        CATALOG,
        "run_backtest",
        {GOLD_BACKTEST_METRICS: [metric_row().as_row()]},
    )

    merges = [text for text in spark.merge_statements() if "backtest_metrics" in text]

    assert len(merges) == 1
    assert f"MERGE INTO {CATALOG}.gold.backtest_metrics" in merges[0]
    assert "ON t.origin_date = s.origin_date AND t.ticker = s.ticker AND t.model = s.model" in merges[0]
    assert "WHEN MATCHED THEN UPDATE SET *" in merges[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in merges[0]


def test_write_gold_never_issues_a_bare_insert():
    spark = FakeSpark()

    write_gold(
        spark,
        CATALOG,
        "run_backtest",
        {
            GOLD_BACKTEST_METRICS: [metric_row().as_row()],
            GOLD_BACKTEST_SUMMARY: [pooled_row().as_row(GENERATED_AT)],
        },
    )

    assert not any(
        re.match(r"\s*INSERT\s+INTO", statement, re.IGNORECASE) for statement in spark.statements
    )
    assert len(spark.merge_statements()) == 3  # two gold tables plus the ledger row


def test_write_gold_writes_one_ledger_row_for_the_whole_task():
    """One audit row per task per run (A-4), even when the task writes several tables."""
    spark = FakeSpark()

    summary = write_gold(
        spark,
        CATALOG,
        "run_backtest",
        {
            GOLD_BACKTEST_METRICS: [metric_row().as_row()],
            GOLD_BACKTEST_SUMMARY: [pooled_row().as_row(GENERATED_AT)],
        },
    )
    rows = spark.ledger_rows()

    assert len(rows) == 1
    assert rows[0]["task"] == "run_backtest"
    assert rows[0]["status"] == STATUS_SUCCEEDED
    assert rows[0]["rows_written"] == 2
    assert rows[0]["error"] is None
    assert rows[0]["finished_at"] >= rows[0]["started_at"]
    assert summary["run_id"] == rows[0]["run_id"]
    assert summary["rows_total"] == 2


def test_write_gold_records_a_failed_ledger_row_and_reraises():
    spark = FakeSpark(fail_on="MERGE INTO")

    with pytest.raises(RuntimeError, match="simulated Spark failure"):
        write_gold(spark, CATALOG, "run_forecasts", {GOLD_REGIME_STATES: [_regime_row()]})

    rows = spark.ledger_rows()

    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_FAILED
    assert rows[0]["error"].startswith("RuntimeError: boom")


def test_write_gold_fails_with_an_actionable_message_when_the_table_is_missing():
    spark = FakeSpark()
    spark.missing_tables.add(f"{CATALOG}.gold.regime_states")

    with pytest.raises(RuntimeError, match="create_delta_tables.sql"):
        write_gold(spark, CATALOG, "fit_models", {GOLD_REGIME_STATES: [_regime_row()]})

    assert spark.ledger_rows()[0]["status"] == STATUS_FAILED


def test_write_gold_rejects_an_unknown_table():
    spark = FakeSpark()

    with pytest.raises(ValueError, match="unknown gold table"):
        write_gold(spark, CATALOG, "fit_models", {"gold.vibes": [{}]})


def test_write_gold_validates_every_table_before_writing_any_of_them():
    """A bad row in the second table must not leave the first one already updated.

    Delta gives no transaction across tables, so the only defence is to validate everything first.
    Half-written gold is worse than a clean failure: it looks like success in whichever table an
    operator happens to check.
    """
    spark = FakeSpark()
    broken = pooled_row().as_row(GENERATED_AT)
    broken.pop("fallback_rate")

    with pytest.raises(ValueError, match="missing \\['fallback_rate'\\]"):
        write_gold(
            spark,
            CATALOG,
            "run_backtest",
            {
                GOLD_BACKTEST_METRICS: [metric_row().as_row()],
                GOLD_BACKTEST_SUMMARY: [broken],
            },
        )

    # The ledger row is the only MERGE that may have run: the good table was never touched.
    assert all("gold." not in statement for statement in spark.merge_statements())
    assert spark.ledger_rows()[0]["status"] == STATUS_FAILED


def test_write_gold_rejects_an_unexpected_column():
    spark = FakeSpark()
    row = metric_row().as_row() | {"sharpe": 1.4}

    with pytest.raises(ValueError, match="unexpected \\['sharpe'\\]"):
        write_gold(spark, CATALOG, "run_backtest", {GOLD_BACKTEST_METRICS: [row]})


# ------------------------------------------------------------------ the rows themselves


def _regime_row(**overrides) -> dict:
    from types import SimpleNamespace

    sorted_params = SimpleNamespace(
        prob_low_vol=0.27,
        prob_high_vol=0.73,
        mus=(0.0004, -0.0015),
        sigmas=(0.0102, 0.0298),
    )
    row = regime_state_row(
        sorted_params,
        ticker="NVDA",
        as_of_date=AS_OF,
        current_news_signal=-0.42,
        model_used="news_markov",
        model_version="b-1",
    )
    return row | overrides


def summary_for(model_used: str, **overrides):
    from src.models.monte_carlo import ForecastSummary

    summary = ForecastSummary(
        model_used=model_used,
        model_version="b-1",
        horizon_days=5,
        n_paths=5000,
        seed=42,
        current_price=180.0,
        price_p10=170.0,
        price_p50=181.0,
        price_p90=193.0,
        return_p10=-0.0556,
        return_p50=0.0056,
        return_p90=0.0722,
        prob_positive=0.53,
        prob_loss_gt_5pct=0.11,
        prob_low_vol=0.27,
        prob_high_vol=0.73,
    )
    return replace(summary, **overrides) if overrides else summary


@pytest.mark.parametrize(
    ("table", "row"),
    [
        (GOLD_REGIME_STATES, _regime_row()),
        (GOLD_FORECAST_RUNS, forecast_run_row(summary_for("markov"), ticker="NVDA", as_of_date=AS_OF)),
        (GOLD_BACKTEST_METRICS, metric_row().as_row()),
        (GOLD_BACKTEST_SUMMARY, pooled_row().as_row(GENERATED_AT)),
    ],
)
def test_every_row_builder_produces_exactly_the_tables_columns(table: str, row: dict):
    """A row builder that drifts from the DDL writes silent NULLs, so pin the two together."""
    assert set(row) == set(GOLD_TABLES[table].columns)


def test_the_forecast_id_is_stable_across_runs():
    """Re-running a day must update the row in place, and anything referencing it must still resolve."""
    first = forecast_id_for("NVDA", AS_OF, "news_markov")

    assert first == forecast_id_for("NVDA", AS_OF, "news_markov")
    assert first != forecast_id_for("NVDA", AS_OF, "markov")
    assert first != forecast_id_for("MSFT", AS_OF, "news_markov")
    assert first != forecast_id_for("NVDA", date(2026, 8, 6), "news_markov")


def test_the_forecast_row_carries_the_identifiers_the_simulation_does_not_own():
    row = forecast_run_row(
        summary_for("news_markov"),
        ticker="NVDA",
        as_of_date=AS_OF,
        generated_at=GENERATED_AT,
    )

    assert row["forecast_id"] == forecast_id_for("NVDA", AS_OF, "news_markov")
    assert row["ticker"] == "NVDA"
    assert row["as_of_date"] == AS_OF
    assert row["generated_at"] == GENERATED_AT
    assert row["model_used"] == "news_markov"
    assert row["return_p50"] == pytest.approx(0.0056)


def test_the_forecast_row_stamps_a_utc_generated_at_when_none_is_given():
    row = forecast_run_row(summary_for("gbm"), ticker="NVDA", as_of_date=AS_OF)

    assert row["generated_at"].tzinfo is not None


def test_a_model_a_forecast_stores_null_regime_probabilities():
    """A model without regimes has no regime probability, and NULL says so — not 0.0."""
    row = forecast_run_row(
        summary_for("gbm", prob_low_vol=None, prob_high_vol=None),
        ticker="NVDA",
        as_of_date=AS_OF,
    )

    assert row["prob_low_vol"] is None
    assert row["prob_high_vol"] is None


def test_the_regime_row_stores_the_decimal_scale_low_vol_first():
    """Gold is decimal throughout, and index 0 IS the calm regime after the mandatory re-sort."""
    row = _regime_row()

    assert row["low_vol_sigma"] == pytest.approx(0.0102)
    assert row["high_vol_sigma"] == pytest.approx(0.0298)
    assert row["low_vol_sigma"] < row["high_vol_sigma"]
    assert row["prob_low_vol"] + row["prob_high_vol"] == pytest.approx(1.0)


def test_a_backtest_row_keeps_the_evidence_behind_its_scores():
    """A Brier score with no record of the probability and the outcome cannot be audited."""
    row = metric_row().as_row()

    assert row["model"] == "news_markov"
    assert row["model_used"] == "markov"  # the fallback is recorded, not hidden
    assert row["failure_reason"].startswith("news_markov:")
    assert row["prob_positive"] == pytest.approx(0.54)
    assert row["realized_return"] == pytest.approx(0.004)
    assert row["covered_80"] is True


def test_a_gbm_backtest_row_reports_no_optimizer_verdict():
    """``converged`` is NULL for GBM: there is no optimizer, and True would claim one succeeded."""
    row = replace(metric_row(), model="gbm", model_used="gbm", converged=None).as_row()

    assert row["converged"] is None


def test_the_ledger_documents_the_gold_task_names():
    """The ``task`` column comment is the only place an operator learns the vocabulary."""
    ddl = _read_ddl()

    for task in ("fit_models", "run_forecasts", "run_backtest"):
        assert task in ddl


# ------------------------------------------------------------------ the backtest runner


def load_backtest_runner():
    """Import ``notebooks/10_backtest_run.py`` by path.

    By path because the filename starts with a digit and cannot be imported normally. The notebook
    is a thin wrapper (spec C-a), but it is the ONLY thing that joins ``run_backtest`` to
    ``write_gold``, so the join is worth a test even though the notebook itself holds no logic.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "notebooks" / "10_backtest_run.py"
    spec = importlib.util.spec_from_file_location("backtest_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__", so _cli() does not run
    return module


class FakeSparkWithFeatures(FakeSpark):
    """A fake session that also answers the one ``daily_features`` read the notebook makes."""

    def __init__(self, frames: dict):
        super().__init__()
        self.frames_by_ticker = frames
        self.requested: list[str] = []

    def sql(self, text: str, args: dict | None = None):
        if "daily_features" in text and text.lstrip().startswith("SELECT"):
            self.statements.append(text)
            ticker = (args or {})["ticker"]
            self.requested.append(ticker)
            return _PandasResult(self.frames_by_ticker[ticker])
        return super().sql(text, args)


class _PandasResult:
    def __init__(self, frame):
        self._frame = frame

    def toPandas(self):  # noqa: N802 — Spark's API
        return self._frame


@pytest.fixture(scope="module")
def runner():
    return load_backtest_runner()


def test_the_runner_resolves_a_ticker_subset(runner):
    """A two-ticker validation run has to be expressible, and ``all`` has to mean the seed set."""
    config = {"tickers": {"seed": ["NVDA", "MSFT", "TSLA"]}}

    assert runner.resolve_tickers(config, "all") == ["MSFT", "NVDA", "TSLA"]
    assert runner.resolve_tickers(config, "") == ["MSFT", "NVDA", "TSLA"]
    assert runner.resolve_tickers(config, "nvda, msft") == ["MSFT", "NVDA"]
    # A watchlist ticker added through the agent is outside the seed and still legitimate.
    assert runner.resolve_tickers(config, "AMD") == ["AMD"]


def test_the_runner_overrides_n_weeks_without_touching_the_config(runner):
    """The file on disk stays the record of what the project ships."""
    from tests.conftest import backtest_config

    config = backtest_config(n_weeks=26)

    assert runner.apply_overrides(config, "4")["backtest"]["n_weeks"] == 4
    assert runner.apply_overrides(config, "")["backtest"]["n_weeks"] == 26
    assert config["backtest"]["n_weeks"] == 26


def test_the_runner_reads_features_for_one_ticker_ordered(runner, backtest_frame):
    """One ``.toPandas()`` per ticker, ordered, projected, ticker as a parameter marker."""
    spark = FakeSparkWithFeatures({"NVDA": backtest_frame})

    frame = runner.read_features(spark, CATALOG, "NVDA")

    assert len(frame) == len(backtest_frame)
    assert spark.requested == ["NVDA"]
    assert "ORDER BY trade_date" in spark.statements[0]
    assert ":ticker" in spark.statements[0]  # never interpolated into the SQL text


def test_the_runner_writes_all_three_gold_tables_under_one_ledger_row(runner, backtest_frame):
    """The end-to-end wiring: features in, gold rows out, one audit row for the task."""
    from tests.conftest import backtest_config

    config = backtest_config(min_train_days=60, n_weeks=1, n_paths=100)
    config["tickers"] = {"seed": ["NVDA"]}
    spark = FakeSparkWithFeatures({"NVDA": backtest_frame})

    summary = runner.main(spark, config, ["NVDA"])

    ledger = spark.ledger_rows()

    assert summary["task"] == "run_backtest"
    assert summary["rows_written"] == {
        GOLD_BACKTEST_METRICS: 3,  # 1 origin x 3 arms
        GOLD_BACKTEST_SUMMARY: 3,  # one pooled row per arm
        GOLD_FORECAST_RUNS: 1,  # the production ladder's forecast, one row per ticker
    }
    assert len(ledger) == 1
    assert ledger[0]["status"] == STATUS_SUCCEEDED
    assert ledger[0]["rows_written"] == summary["rows_total"] == 7


def test_the_runner_refuses_to_score_a_ticker_with_no_features(runner):
    """A missing feature build must stop the run, not silently produce an empty backtest."""
    import pandas as pd

    from tests.conftest import backtest_config

    empty = pd.DataFrame(
        {"trade_date": [], "close": [], "log_return": [], "news_sentiment_3d": []}
    )
    spark = FakeSparkWithFeatures({"NVDA": empty})

    with pytest.raises(SystemExit, match="no features"):
        runner.main(spark, backtest_config(), ["NVDA"])
