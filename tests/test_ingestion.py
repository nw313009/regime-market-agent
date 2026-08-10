"""Bronze ingestion tests (spec A-2). Pure functions only — no SparkSession, no network.

What is covered here:

- the incremental watermark logic, both branches: empty table -> configured backfill window,
  populated table -> since that ticker's own last stored bar/article
- per-source timestamp parsing: aggregates epoch-milliseconds vs news ISO-8601 UTC strings
- the news explode FROM insights, including the strict-subset rule that a ticker listed in
  ``tickers`` with no insight yields NO row
- row builders staying aligned with the table DDL column lists

What is deliberately NOT covered here: MERGE semantics. See the integration test note at the
bottom of this file — those assertions need real Delta, and a fake local Spark would prove
nothing about a MERGE.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.ingestion import (
    RUNS_COLUMNS,
    RunRecord,
    dedupe_by_keys,
    qualified,
    resolve_universe,
    rows_to_tuples,
    truncate_error,
)
from src.ingestion.ingest_news import (
    NEWS_COLUMNS,
    build_news_rows,
    news_fetch_start,
    parse_published_utc,
)
from src.ingestion.ingest_prices import (
    PRICE_COLUMNS,
    build_price_rows,
    epoch_ms_to_session_date,
    epoch_ms_to_utc,
    price_fetch_window,
)
from src.ingestion.massive_client import REQUEST_ID_KEY, SOURCE
from tests.conftest import STRICT_SUBSET_ARTICLE_ID, UNKNOWN_LABEL_ARTICLE_ID

BACKFILL_START = date(2024, 8, 1)
TODAY = date(2026, 8, 9)
INGESTED_AT = datetime(2026, 8, 9, 22, 30, tzinfo=timezone.utc)
CONFIG = {
    "catalog": "market_intel",
    "tickers": {"seed": ["NVDA", "MSFT", "TSLA", "AMZN", "GOOGL"]},
    "massive": {"backfill_start_date": "2024-08-01"},
}


# ============================================================ prices: incremental window


def test_empty_table_uses_the_configured_backfill_window():
    start, end = price_fetch_window(None, BACKFILL_START, TODAY)

    assert (start, end) == (BACKFILL_START, TODAY)


def test_populated_table_fetches_from_the_last_stored_session():
    # 2026-07-06 04:00Z is 2026-07-06 00:00 in New York — the session the bar belongs to.
    watermark = datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc)

    start, end = price_fetch_window(watermark, BACKFILL_START, TODAY)

    # Inclusive of the watermark session: one re-fetched bar per ticker per run repairs a
    # session that was ingested mid-day, and the MERGE absorbs the duplicate.
    assert (start, end) == (date(2026, 7, 6), TODAY)


def test_watermark_is_read_in_exchange_time_not_utc():
    # 2026-07-07 03:00Z is still 2026-07-06 23:00 in New York, so the session is the 6th.
    watermark = datetime(2026, 7, 7, 3, 0, tzinfo=timezone.utc)

    start, _ = price_fetch_window(watermark, BACKFILL_START, TODAY)

    assert start == date(2026, 7, 6)


def test_naive_watermark_is_treated_as_wall_clock_date():
    start, _ = price_fetch_window(datetime(2026, 7, 6, 4, 0), BACKFILL_START, TODAY)

    assert start == date(2026, 7, 6)


def test_date_watermark_is_accepted_as_is():
    start, _ = price_fetch_window(date(2026, 7, 6), BACKFILL_START, TODAY)

    assert start == date(2026, 7, 6)


@pytest.mark.parametrize(
    "watermark",
    [datetime(2026, 12, 31, 5, 0, tzinfo=timezone.utc), None],
    ids=["future watermark", "future backfill date"],
)
def test_window_is_clamped_to_today(watermark):
    backfill = date(2027, 1, 1) if watermark is None else BACKFILL_START

    start, end = price_fetch_window(watermark, backfill, TODAY)

    assert start == TODAY
    assert end == TODAY
    assert start <= end  # an inverted range would be a wasted request at best


# ================================================================= prices: timestamps


@pytest.mark.parametrize(
    "epoch_ms,expected",
    [
        (1782878400000, date(2026, 7, 1)),
        (1782964800000, date(2026, 7, 2)),
        (1783310400000, date(2026, 7, 6)),
    ],
)
def test_epoch_ms_maps_to_the_exchange_session_date_in_summer(epoch_ms, expected):
    # Verified live values: daily bars are stamped 04:00Z during EDT, i.e. 00:00 New York.
    assert epoch_ms_to_session_date(epoch_ms) == expected


def test_epoch_ms_maps_to_the_exchange_session_date_in_winter():
    # During EST the same session opens at 05:00Z. Taking the UTC date happens to agree here,
    # but the conversion is explicit so it stays correct if the vendor shifts the stamp.
    winter_bar = datetime(2026, 1, 5, 5, 0, tzinfo=timezone.utc)
    epoch_ms = int(winter_bar.timestamp() * 1000)

    assert epoch_ms_to_session_date(epoch_ms) == date(2026, 1, 5)
    assert winter_bar.astimezone(ZoneInfo("America/New_York")).hour == 0


def test_epoch_ms_to_utc_is_timezone_aware():
    parsed = epoch_ms_to_utc(1782878400000)

    assert parsed.tzinfo is not None
    assert parsed == datetime(2026, 7, 1, 4, 0, tzinfo=timezone.utc)


# ================================================================== prices: row builder


def test_build_price_rows_maps_the_verified_payload(aggregates_results):
    for bar in aggregates_results:
        bar[REQUEST_ID_KEY] = "rid-aggs"

    rows = build_price_rows("NVDA", aggregates_results, ingested_at=INGESTED_AT)

    assert len(rows) == 3
    first = rows[0]
    assert first["ticker"] == "NVDA"
    assert first["open"] == 196.2
    assert first["high"] == 199.85
    assert first["low"] == 193.45
    assert first["close"] == 197.58
    assert first["vwap"] == 197.0727
    assert first["volume"] == pytest.approx(1.46147597081851e08)
    assert first["transactions"] == 2330312
    assert first["t_epoch_ms"] == 1782878400000
    assert first["source_timestamp"] == datetime(2026, 7, 1, 4, 0, tzinfo=timezone.utc)
    assert first["source"] == SOURCE
    assert first["ingested_at"] == INGESTED_AT
    assert first["request_id"] == "rid-aggs"


def test_build_price_rows_matches_the_table_columns(aggregates_results):
    rows = build_price_rows("NVDA", aggregates_results, ingested_at=INGESTED_AT)

    # Catches a column added to the DDL but forgotten in the row builder, which would otherwise
    # surface as an opaque createDataFrame schema error on the cluster.
    assert set(rows[0]) == set(PRICE_COLUMNS)


def test_build_price_rows_skips_a_bar_without_a_timestamp(caplog):
    with caplog.at_level(logging.WARNING):
        rows = build_price_rows("NVDA", [{"o": 1.0, "c": 2.0}], ingested_at=INGESTED_AT)

    assert rows == []
    assert "without t" in caplog.text


def test_build_price_rows_keeps_volume_as_float(aggregates_results):
    rows = build_price_rows("NVDA", aggregates_results, ingested_at=INGESTED_AT)

    assert all(isinstance(row["volume"], float) for row in rows)
    assert all(isinstance(row["transactions"], int) for row in rows)


# ================================================================== news: incremental


def test_news_empty_table_uses_the_configured_backfill_window():
    assert news_fetch_start(None, BACKFILL_START) == "2024-08-01T00:00:00Z"


def test_news_populated_table_fetches_from_the_last_stored_article():
    watermark = datetime(2026, 8, 10, 2, 15, tzinfo=timezone.utc)

    # Inclusive: an article published in the same second as the watermark must not fall through
    # the gap, and (article_id, ticker) makes the overlap idempotent.
    assert news_fetch_start(watermark, BACKFILL_START) == "2026-08-10T02:15:00Z"


def test_news_watermark_in_another_timezone_is_converted_to_utc():
    watermark = datetime(2026, 8, 9, 22, 15, tzinfo=ZoneInfo("America/New_York"))

    assert news_fetch_start(watermark, BACKFILL_START) == "2026-08-10T02:15:00Z"


def test_news_naive_watermark_is_treated_as_utc():
    assert news_fetch_start(datetime(2026, 8, 10, 2, 15), BACKFILL_START) == "2026-08-10T02:15:00Z"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-08-10T02:15:00Z", datetime(2026, 8, 10, 2, 15, tzinfo=timezone.utc)),
        ("2026-08-10T02:15:00+00:00", datetime(2026, 8, 10, 2, 15, tzinfo=timezone.utc)),
        ("2026-08-10T02:15:00.123Z", datetime(2026, 8, 10, 2, 15, 0, 123000, tzinfo=timezone.utc)),
        ("2026-08-09T22:15:00-04:00", datetime(2026, 8, 10, 2, 15, tzinfo=timezone.utc)),
    ],
)
def test_parse_published_utc_handles_the_iso_forms(raw, expected):
    assert parse_published_utc(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "not-a-timestamp", "1782878400000"])
def test_parse_published_utc_returns_none_for_unusable_values(raw):
    assert parse_published_utc(raw) is None


# =================================================================== news: the explode


def test_news_explode_comes_from_insights_not_tickers(news_results):
    article = next(a for a in news_results if a["id"] == STRICT_SUBSET_ARTICLE_ID)

    rows = build_news_rows([article], ingested_at=INGESTED_AT)

    # tickers = [SNDK, NVDA, TSLA, WDC]; insights = [SNDK, NVDA]. TSLA and WDC have no insight,
    # so they get NO row. This is the A-3 rule, and the assertion that fails loudly if someone
    # "simplifies" the explode back to the tickers array.
    assert {row["ticker"] for row in rows} == {"SNDK", "NVDA"}
    assert "TSLA" not in {row["ticker"] for row in rows}
    assert "WDC" not in {row["ticker"] for row in rows}


def test_news_rows_keep_the_full_raw_tickers_array(news_results):
    article = next(a for a in news_results if a["id"] == STRICT_SUBSET_ARTICLE_ID)

    rows = build_news_rows([article], ingested_at=INGESTED_AT)

    # The dropped tickers stay auditable even though they produced no rows.
    assert rows[0]["article_tickers"] == ["SNDK", "NVDA", "TSLA", "WDC"]


def test_news_rows_take_ticker_and_sentiment_from_the_same_insight(news_results):
    article = next(a for a in news_results if a["id"] == STRICT_SUBSET_ARTICLE_ID)

    by_ticker = {row["ticker"]: row for row in build_news_rows([article], ingested_at=INGESTED_AT)}

    assert by_ticker["SNDK"]["sentiment"] == "positive"
    assert by_ticker["NVDA"]["sentiment"] == "neutral"
    assert "3,000% surge" in by_ticker["SNDK"]["sentiment_reasoning"]


def test_news_rows_map_the_verified_fields(news_results):
    article = next(a for a in news_results if a["id"] == UNKNOWN_LABEL_ARTICLE_ID)
    article[REQUEST_ID_KEY] = "rid-news"

    row = build_news_rows([article], ingested_at=INGESTED_AT)[0]

    assert row["article_id"] == UNKNOWN_LABEL_ARTICLE_ID
    assert row["ticker"] == "NVDA"
    assert row["publisher_name"] == "Benzinga"  # publisher is nested; the mapping takes .name
    assert row["publisher_homepage_url"] == "https://www.benzinga.com/"
    assert row["published_utc"] == "2026-08-09T13:45:30Z"
    assert row["source_timestamp"] == datetime(2026, 8, 9, 13, 45, 30, tzinfo=timezone.utc)
    assert row["title"].startswith("Nvidia Slips After Guidance")
    assert row["author"] == "Staff Writer"
    assert row["keywords"] == ["earnings"]
    assert row["source"] == SOURCE
    assert row["ingested_at"] == INGESTED_AT
    assert row["request_id"] == "rid-news"


def test_news_rows_keep_the_raw_sentiment_label_and_derive_no_score(news_results):
    article = next(a for a in news_results if a["id"] == UNKNOWN_LABEL_ARTICLE_ID)

    row = build_news_rows([article], ingested_at=INGESTED_AT)[0]

    # Bronze stores the vendor's label verbatim, even an unrecognized one. The ±1/0 mapping and
    # its warning belong to the silver build (A-3), not to ingestion.
    assert row["sentiment"] == "mixed"
    assert "sentiment_score" not in row


def test_news_rows_match_the_table_columns(news_results):
    rows = build_news_rows(news_results, ingested_at=INGESTED_AT)

    assert set(rows[0]) == set(NEWS_COLUMNS)


def test_all_fixture_articles_explode_to_the_expected_row_count(news_results):
    rows = build_news_rows(news_results, ingested_at=INGESTED_AT)

    # 2 insights + 1 insight + 1 insight = 4 rows from 3 articles listing 6 ticker mentions.
    assert len(rows) == 4
    assert len({(row["article_id"], row["ticker"]) for row in rows}) == 4  # MERGE keys unique


def test_article_without_insights_produces_no_rows(news_results):
    article = dict(news_results[0], insights=[])

    assert build_news_rows([article], ingested_at=INGESTED_AT) == []


def test_article_without_id_is_skipped_with_a_warning(news_results, caplog):
    article = dict(news_results[0])
    article.pop("id")

    with caplog.at_level(logging.WARNING):
        rows = build_news_rows([article], ingested_at=INGESTED_AT)

    assert rows == []
    assert "without id" in caplog.text


def test_article_with_unparseable_timestamp_is_skipped_with_a_warning(news_results, caplog):
    article = dict(news_results[0], published_utc="yesterday")

    with caplog.at_level(logging.WARNING):
        rows = build_news_rows([article], ingested_at=INGESTED_AT)

    assert rows == []
    assert "unparseable published_utc" in caplog.text


def test_insight_without_a_ticker_is_skipped_with_a_warning(news_results, caplog):
    article = dict(
        news_results[2],
        insights=[{"sentiment": "positive", "sentiment_reasoning": "no ticker on this insight"}],
    )

    with caplog.at_level(logging.WARNING):
        rows = build_news_rows([article], ingested_at=INGESTED_AT)

    assert rows == []
    assert "insight without ticker" in caplog.text


def test_insight_tickers_are_normalized(news_results):
    article = dict(
        news_results[2],
        insights=[{"ticker": " msft ", "sentiment": "positive", "sentiment_reasoning": "x"}],
    )

    rows = build_news_rows([article], ingested_at=INGESTED_AT)

    assert rows[0]["ticker"] == "MSFT"


def test_publisher_string_instead_of_dict_does_not_break_ingestion(news_results):
    article = dict(news_results[2], publisher="Reuters")

    rows = build_news_rows([article], ingested_at=INGESTED_AT)

    assert rows[0]["publisher_name"] == "Reuters"


# ==================================================================== shared helpers


def test_universe_is_the_seed_tickers_when_there_is_no_watchlist():
    assert resolve_universe(CONFIG) == ["AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]


def test_universe_adds_watchlist_tickers_deduplicated_and_sorted():
    # AMD is the CDC demo ticker: it reaches the universe only via the watchlist.
    assert resolve_universe(CONFIG, watchlist=["amd", "NVDA", " AMD "]) == [
        "AMD",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
        "TSLA",
    ]


def test_universe_survives_a_config_without_seed_tickers():
    assert resolve_universe({}, watchlist=["NVDA"]) == ["NVDA"]


def test_dedupe_by_keys_keeps_the_last_row_per_key():
    rows = [
        {"article_id": "a", "ticker": "NVDA", "title": "first"},
        {"article_id": "a", "ticker": "MSFT", "title": "other ticker"},
        {"article_id": "a", "ticker": "NVDA", "title": "second"},
    ]

    deduped = dedupe_by_keys(rows, ("article_id", "ticker"))

    # Delta fails a MERGE when several source rows match one target row, so this is required
    # rather than defensive: a paginated fetch can legitimately repeat an article.
    assert len(deduped) == 2
    assert {row["title"] for row in deduped} == {"second", "other ticker"}


def test_rows_to_tuples_follows_the_column_order():
    rows = [{"b": 2, "a": 1}]

    assert rows_to_tuples(rows, ("a", "b", "missing")) == [(1, 2, None)]


def test_run_record_row_matches_the_ledger_columns():
    run = RunRecord(run_id="r1", task="ingest_prices", started_at=INGESTED_AT)

    row = run.as_row()

    assert set(row) == set(RUNS_COLUMNS)
    assert row["status"] == "succeeded"
    assert row["rows_written"] == 0
    assert row["error"] is None


def test_truncate_error_caps_the_ledger_column():
    truncated = truncate_error("x" * 5000)

    assert len(truncated) == 1000
    assert truncated.endswith("...")


def test_qualified_prefixes_the_catalog_from_config():
    assert qualified("market_intel", "bronze.prices_raw") == "market_intel.bronze.prices_raw"


# =====================================================================================
# TODO (integration, run in the Databricks workspace — not fakeable locally):
#
# These assertions need real Delta and a real SparkSession. A local fake would exercise the
# stub, not the MERGE, and would pass while production duplicated rows.
#
# 1. IDEMPOTENCY (spec A-5 test_idempotency, rule 6). Run ingest_prices.main twice against the
#    same window and assert identical row counts AND identical checksums for
#    bronze.prices_raw; same for ingest_news.main and bronze.news_raw. Row counts alone would
#    miss a MERGE that rewrites rows it should have left alone.
# 2. MERGE KEYS. Re-ingest a window whose last session was already stored and assert the bar
#    count for that (ticker, source_timestamp) stays 1 — this is what proves the key, and it is
#    the assertion that fails if someone switches bronze prices to trade_date.
# 3. WATERMARK ADVANCE. After a successful run, assert max(source_timestamp) per ticker moved
#    forward and that the next run's requested window starts from it (read the logged
#    from=/published_after= values, or assert on request counts).
# 4. LEDGER. Assert exactly one bronze.ingestion_runs row per main() call, status succeeded,
#    rows_written matching the merged count; then force a failure (revoke the secret) and assert
#    a status=failed row exists with finished_at set and a redacted, non-null error that
#    contains neither "apiKey=" nor a response body.
# 5. EMPTY-TABLE BRANCH. Truncate bronze.prices_raw for one ticker and assert the next run
#    requests from massive.backfill_start_date rather than from another ticker's watermark.
# =====================================================================================
