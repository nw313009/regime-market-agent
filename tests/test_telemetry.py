"""Model-call telemetry tests (spec C-3).

Three things are pinned here:

- RECORDING IS BUFFERED and never touches Spark. ``record`` runs inside a chat turn.
- FLUSHING IS A MERGE on ``call_id`` (rule 4), and a failed flush puts its records back so the
  next one retries them — which is only safe because the write is a MERGE.
- THE MODE IS CONFIGURABLE, because the Databricks App has no SparkSession (spec C-5). ``log``
  and ``off`` must work with no ``spark`` argument at all.

The Spark fake is ``tests.conftest.FakeSpark``, the same one the bronze/silver/gold write tests
use: one fake, one contract for every ledgered write path in the repo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.llm import telemetry
from src.llm.telemetry import (
    MAX_BUFFERED_RECORDS,
    MODE_DELTA,
    MODE_LOG,
    MODE_OFF,
    MODEL_CALLS_COLUMNS,
    MODEL_CALLS_KEYS,
    MODEL_CALLS_SCHEMA_DDL,
    MODEL_CALLS_TABLE,
    ModelCall,
    TelemetryModeError,
    buffered,
    clear,
    configure,
    flush,
    record,
    resolve_mode,
)
from tests.conftest import FakeSpark

CATALOG = "market_intel"
FQN = f"{CATALOG}.{MODEL_CALLS_TABLE}"
DDL_PATH = Path(__file__).resolve().parents[1] / "setup" / "create_delta_tables.sql"
CONFIG = {"catalog": CATALOG, "telemetry": {"mode": MODE_DELTA}}


@pytest.fixture(autouse=True)
def reset_telemetry(monkeypatch):
    monkeypatch.delenv(telemetry.MODE_ENV_VAR, raising=False)
    configure(None)
    clear()
    yield
    configure(None)
    clear()


def a_call(task: str = "agent", *, ok: bool = True) -> ModelCall | None:
    return record(
        task=task,
        model="meta-llama-3.3-70b-instruct",
        latency_ms=812.5,
        ok=ok,
        in_tokens=412 if ok else None,
        out_tokens=137 if ok else None,
        config=CONFIG,
    )


# ---------------------------------------------------------------- mode resolution


def test_mode_defaults_to_the_config_file_value():
    assert resolve_mode(CONFIG) == MODE_DELTA


def test_environment_overrides_the_config_file(monkeypatch):
    # One repository config is read by both the jobs and the app, and only the app lacks Spark.
    monkeypatch.setenv(telemetry.MODE_ENV_VAR, "log")

    assert resolve_mode(CONFIG) == MODE_LOG


def test_configure_overrides_everything(monkeypatch):
    monkeypatch.setenv(telemetry.MODE_ENV_VAR, "log")
    configure(MODE_OFF)

    assert resolve_mode(CONFIG) == MODE_OFF


def test_an_unset_mode_falls_back_to_log_only():
    # A default that needs a SparkSession would turn a forgotten setting into an agent failure.
    assert resolve_mode({}) == MODE_LOG


def test_an_unknown_mode_is_rejected():
    with pytest.raises(TelemetryModeError, match="expected one of"):
        resolve_mode({"telemetry": {"mode": "delta-ish"}})

    with pytest.raises(TelemetryModeError):
        configure("everything")


# --------------------------------------------------------------------- buffering


def test_record_buffers_the_seven_spec_fields_plus_an_identity():
    before = datetime.now(timezone.utc)
    call = a_call()

    assert call is not None
    assert call.ts >= before
    assert (call.task, call.ok) == ("agent", True)
    assert (call.in_tokens, call.out_tokens) == (412, 137)
    assert buffered() == (call,)


def test_row_columns_match_the_write_contract():
    row = a_call().as_row()

    # Catches a field added to ModelCall but forgotten in the columns tuple, which merge_rows
    # would otherwise write as a silent NULL.
    assert set(row) == set(MODEL_CALLS_COLUMNS)


def test_every_record_gets_its_own_call_id():
    first, second = a_call(), a_call()

    # The MERGE key. Without a distinct id per record the second call would overwrite the first.
    assert first.call_id != second.call_id
    assert MODEL_CALLS_KEYS == ("call_id",)


def test_failed_calls_are_recorded_with_null_tokens():
    call = a_call(ok=False)

    assert call.ok is False
    assert (call.in_tokens, call.out_tokens) == (None, None)


def test_off_mode_records_nothing():
    configure(MODE_OFF)

    assert a_call() is None
    assert buffered() == ()


def test_the_buffer_is_bounded_and_drops_the_oldest(caplog):
    configure(MODE_LOG)
    with caplog.at_level(logging.WARNING, logger="src.llm.telemetry"):
        for index in range(MAX_BUFFERED_RECORDS + 5):
            a_call(task=f"agent-{index}")

    records = buffered()
    assert len(records) == MAX_BUFFERED_RECORDS
    assert records[-1].task == f"agent-{MAX_BUFFERED_RECORDS + 4}"
    assert records[0].task == "agent-5"  # the first five were dropped, not the last five
    assert "buffer full" in caplog.text


# ------------------------------------------------------------------- flush: delta


def test_flush_merges_the_buffer_into_gold_model_calls():
    configure(MODE_DELTA)
    a_call()
    a_call(task="slm")
    spark = FakeSpark()

    written = flush(spark, config=CONFIG)

    assert written == 2
    assert buffered() == ()

    (statement,) = spark.merge_statements()
    assert f"MERGE INTO {FQN}" in statement
    assert "t.call_id = s.call_id" in statement
    assert "WHEN MATCHED THEN UPDATE SET *" in statement  # a MERGE, never a blind INSERT

    (frame,) = [f for f in spark.frames if f.schema == MODEL_CALLS_SCHEMA_DDL]
    assert len(frame.rows) == 2


def test_flush_takes_the_catalog_from_config_when_not_given():
    configure(MODE_DELTA)
    a_call()
    spark = FakeSpark()

    flush(spark, config=CONFIG)

    assert FQN in spark.merge_statements()[0]


def test_flush_fails_loudly_when_the_table_was_never_created():
    configure(MODE_DELTA)
    a_call()
    spark = FakeSpark()
    spark.missing_tables.add(FQN)

    with pytest.raises(RuntimeError, match="create_delta_tables.sql"):
        flush(spark, config=CONFIG)


def test_a_failed_write_returns_the_records_to_the_buffer():
    configure(MODE_DELTA)
    first = a_call()
    second = a_call(task="slm")
    spark = FakeSpark(fail_on="MERGE INTO")

    with pytest.raises(RuntimeError, match="simulated Spark failure"):
        flush(spark, config=CONFIG)

    # Re-queued in order, so the next flush retries them. Safe only because the write is a MERGE
    # on call_id: a partially applied flush cannot duplicate a row.
    assert [c.call_id for c in buffered()] == [first.call_id, second.call_id]


def test_delta_mode_without_a_spark_session_is_an_explicit_error():
    configure(MODE_DELTA)
    call = a_call()

    with pytest.raises(TelemetryModeError, match="TELEMETRY_MODE=log"):
        flush(config=CONFIG)

    assert buffered() == (call,)  # nothing is lost by the refusal


# --------------------------------------------------------------- flush: log / off


def test_log_mode_flushes_without_spark_and_writes_nothing(caplog):
    configure(MODE_LOG)
    a_call()

    with caplog.at_level(logging.INFO, logger="src.llm.telemetry"):
        drained = flush(config=CONFIG)

    assert drained == 1
    assert buffered() == ()
    assert "task=agent" in caplog.text
    assert "written=0" in caplog.text


def test_off_mode_flush_is_a_no_op():
    configure(MODE_OFF)

    assert flush(config=CONFIG) == 0


def test_flushing_an_empty_buffer_needs_no_spark():
    configure(MODE_DELTA)

    assert flush(config=CONFIG) == 0


# ------------------------------------------------------- the DDL and the code agree


def test_the_delta_ddl_declares_exactly_the_recorded_columns():
    text = DDL_PATH.read_text(encoding="utf-8")
    marker = f"CREATE TABLE IF NOT EXISTS {FQN} ("
    assert marker in text, f"no CREATE TABLE for {MODEL_CALLS_TABLE} in {DDL_PATH.name}"

    block = text[text.index(marker) + len(marker) : text.index(";", text.index(marker))]
    declared = [
        line.strip().split()[0]
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith(("--", ")", "USING", "COMMENT"))
    ]

    assert declared == list(MODEL_CALLS_COLUMNS)


# --------------------------------------------------------------------- integration
# The real call_model, the real telemetry module and the real write layer, with only the two
# boundaries this machine does not have faked out: HTTP and Spark.


def test_a_model_call_lands_in_gold_model_calls_end_to_end():
    from tests.test_call_model import CONFIG as MODEL_CONFIG
    from tests.test_call_model import CREDS, MESSAGES, FakeSession, completion
    from src.llm.call_model import call_model

    configure(MODE_DELTA)
    session = FakeSession(responses=[completion()])

    response = call_model("agent", MESSAGES, config=MODEL_CONFIG, session=session, credentials=CREDS)
    spark = FakeSpark()
    written = flush(spark, catalog=CATALOG)

    assert written == 1
    (frame,) = [f for f in spark.frames if f.schema == MODEL_CALLS_SCHEMA_DDL]
    row = dict(zip(MODEL_CALLS_COLUMNS, frame.rows[0]))
    assert row["task"] == "agent"
    assert row["model"] == response.model
    assert row["ok"] is True
    assert (row["in_tokens"], row["out_tokens"]) == (response.in_tokens, response.out_tokens)
    assert row["latency_ms"] == pytest.approx(response.latency_ms)
