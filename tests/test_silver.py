"""Silver build tests (spec A-3). Pure functions and generated SQL only - no SparkSession.

What is covered here:

- the sentiment mapping, including an unrecognized label degrading to 0 with a logged warning
- epoch-milliseconds to America/New_York trading date, including both DST boundaries
- embedding_text assembly with a missing, blank or whitespace-only description
- the generated SQL agreeing with the Python rule it mirrors, so the two cannot drift
- the CDF property being present on silver.news_articles at CREATE time

What is deliberately NOT covered here: MERGE semantics and double-run idempotency. See the
integration test note at the bottom, consistent with A-2 - a fake local Spark would exercise the
stub, not the MERGE.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.ingest_prices import epoch_ms_to_session_date
from src.pipelines import latest_per_key_sql, merge_sql
from src.pipelines.silver_news import (
    EMBEDDING_TEXT_EXPR,
    EMBEDDING_TEXT_SEPARATOR,
    NEWS_ARTICLE_COLUMNS,
    PUBLISHED_AT_EXPR,
    SENTIMENT_SCORES,
    UNKNOWN_SENTIMENT_SCORE,
    build_embedding_text,
    normalize_sentiment_label,
    sentiment_score,
    sentiment_score_sql,
    unknown_sentiment_labels,
)
from src.pipelines.silver_news import MERGE_KEYS as NEWS_MERGE_KEYS
from src.pipelines.silver_news import build_source_sql as build_news_source_sql
from src.pipelines import EXCHANGE_TZ_NAME
from src.pipelines.silver_prices import (
    DAILY_PRICE_COLUMNS,
    TRADE_DATE_EXPR,
    VOLUME_EXPR,
    trade_date_from_epoch_ms,
    trade_date_from_utc,
    volume_to_long,
)
from src.pipelines.silver_prices import MERGE_KEYS as PRICE_MERGE_KEYS
from src.pipelines.silver_prices import build_source_sql as build_price_source_sql
from tests.conftest import UNKNOWN_LABEL_ARTICLE_ID

CATALOG = "market_intel"
DDL_PATH = Path(__file__).resolve().parents[1] / "setup" / "create_delta_tables.sql"


def _ddl_block(table: str) -> str:
    """The CREATE TABLE statement for ``table``, from the real DDL file."""
    text = DDL_PATH.read_text(encoding="utf-8")
    marker = f"CREATE TABLE IF NOT EXISTS market_intel.{table} ("
    assert marker in text, f"no CREATE TABLE for {table} in {DDL_PATH.name}"
    start = text.index(marker)
    return text[start : text.index(";", start) + 1]


# ==================================================================== sentiment score


@pytest.mark.parametrize(
    "label,expected",
    [("positive", 1), ("neutral", 0), ("negative", -1)],
)
def test_recognized_labels_map_to_their_scores(label, expected):
    assert sentiment_score(label) == expected


@pytest.mark.parametrize("label", ["Positive", " POSITIVE ", "pOsItIvE"])
def test_label_matching_ignores_case_and_whitespace(label):
    assert sentiment_score(label) == 1


def test_unknown_label_degrades_to_zero_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        score = sentiment_score("mixed")

    # A-3: a new vendor label must degrade to neutral LOUDLY - never fail the build, never get
    # silently dropped.
    assert score == 0
    assert "unrecognized sentiment label" in caplog.text
    assert "mixed" in caplog.text


def test_the_fixture_article_with_an_unknown_label_scores_zero(news_results, caplog):
    article = next(a for a in news_results if a["id"] == UNKNOWN_LABEL_ARTICLE_ID)

    with caplog.at_level(logging.WARNING):
        score = sentiment_score(article["insights"][0]["sentiment"])

    assert score == UNKNOWN_SENTIMENT_SCORE
    assert "unrecognized sentiment label" in caplog.text


@pytest.mark.parametrize("label", [None, "", "   "])
def test_missing_label_scores_zero_without_a_warning(label, caplog):
    with caplog.at_level(logging.WARNING):
        score = sentiment_score(label)

    # Not new vocabulary, so warning here would bury the case that matters.
    assert score == 0
    assert caplog.text == ""


def test_normalize_sentiment_label():
    assert normalize_sentiment_label("  NEGATIVE ") == "negative"
    assert normalize_sentiment_label("") is None
    assert normalize_sentiment_label(None) is None


def test_unknown_sentiment_labels_reports_only_new_vocabulary():
    labels = ["positive", "POSITIVE", "mixed", "Mixed", "speculative", None, "", "neutral"]

    assert unknown_sentiment_labels(labels) == ["mixed", "speculative"]


def test_unknown_sentiment_labels_is_empty_for_a_clean_source():
    assert unknown_sentiment_labels(["positive", "neutral", "negative"]) == []


# ------------------------------------------------- generated SQL mirrors the Python rule


def test_sentiment_case_expression_is_generated_from_the_mapping():
    generated = sentiment_score_sql()

    for label, score in SENTIMENT_SCORES.items():
        assert f"WHEN '{label}' THEN {score}" in generated
    assert f"ELSE {UNKNOWN_SENTIMENT_SCORE} END" in generated


def test_sentiment_sql_and_python_recognize_exactly_the_same_labels():
    in_sql = set(re.findall(r"WHEN '([a-z]+)' THEN", sentiment_score_sql()))

    # The guard against a label being added to one implementation only.
    assert in_sql == set(SENTIMENT_SCORES)


def test_sentiment_sql_normalizes_the_label_like_python_does():
    assert "lower(trim(sentiment))" in sentiment_score_sql()


def test_sentiment_sql_takes_a_column_override():
    assert sentiment_score_sql("s.sentiment").startswith("CASE lower(trim(s.sentiment))")


# =================================================================== embedding_text


def test_embedding_text_joins_title_and_description_with_a_newline():
    assert build_embedding_text("Title", "Description") == "Title\nDescription"


@pytest.mark.parametrize("description", [None, "", "   "])
def test_missing_description_yields_the_title_with_no_trailing_separator(description):
    result = build_embedding_text("Title", description)

    # A trailing newline would be embedded as content by the AI Search index.
    assert result == "Title"
    assert not result.endswith(EMBEDDING_TEXT_SEPARATOR)


def test_missing_title_yields_the_description_alone():
    assert build_embedding_text(None, "Description") == "Description"


def test_missing_title_and_description_yields_none():
    assert build_embedding_text(None, None) is None
    assert build_embedding_text("  ", "") is None


def test_embedding_text_trims_each_part():
    assert build_embedding_text("  Title  ", "  Description  ") == "Title\nDescription"


def test_embedding_text_uses_the_real_fixture_article(news_results):
    article = news_results[2]

    result = build_embedding_text(article["title"], article["description"])

    assert result == f"{article['title']}\n{article['description']}"


def test_embedding_text_sql_mirrors_the_python_rule():
    # concat_ws skips NULLs, nullif(trim(x), '') turns a blank into NULL, and the outer nullif
    # returns NULL rather than concat_ws's empty string when both parts are missing.
    assert r"concat_ws('\n'" in EMBEDDING_TEXT_EXPR
    assert "nullif(trim(title), '')" in EMBEDDING_TEXT_EXPR
    assert "nullif(trim(description), '')" in EMBEDDING_TEXT_EXPR
    assert EMBEDDING_TEXT_EXPR.startswith("nullif(concat_ws")


# ====================================================================== trade_date


@pytest.mark.parametrize(
    "epoch_ms,expected",
    [
        (1782878400000, date(2026, 7, 1)),
        (1782964800000, date(2026, 7, 2)),
        (1783310400000, date(2026, 7, 6)),
    ],
)
def test_verified_bars_map_to_their_session_date(epoch_ms, expected):
    assert trade_date_from_epoch_ms(epoch_ms) == expected


@pytest.mark.parametrize(
    "utc_hour,session_date",
    [
        # US DST 2026 starts Sunday 2026-03-08 and ends Sunday 2026-11-01. A daily bar is stamped
        # 00:00 New York, which is 04:00Z under EDT and 05:00Z under EST.
        (4, date(2026, 3, 9)),  # first EDT session
        (5, date(2026, 3, 6)),  # last EST session, stamped an hour later
    ],
)
def test_spring_forward_boundary(utc_hour, session_date):
    stamped = datetime(session_date.year, session_date.month, session_date.day, utc_hour, tzinfo=timezone.utc)

    assert trade_date_from_utc(stamped) == session_date


@pytest.mark.parametrize(
    "utc_hour,session_date",
    [
        (4, date(2026, 10, 30)),  # last EDT session
        (5, date(2026, 11, 2)),  # first EST session after the fall-back
    ],
)
def test_fall_back_boundary(utc_hour, session_date):
    stamped = datetime(session_date.year, session_date.month, session_date.day, utc_hour, tzinfo=timezone.utc)

    assert trade_date_from_utc(stamped) == session_date


def test_a_winter_instant_at_0400z_belongs_to_the_previous_session():
    # 04:00Z in November is 23:00 the previous day in New York. Truncating the UTC date would
    # answer 2026-11-02; the exchange-timezone rule answers 2026-11-01. This is the exact bug the
    # rule exists to prevent, which is why it is asserted rather than assumed.
    stamped = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)

    assert stamped.date() == date(2026, 11, 2)
    assert trade_date_from_utc(stamped) == date(2026, 11, 1)


def test_naive_instants_are_treated_as_utc():
    assert trade_date_from_utc(datetime(2026, 7, 1, 4, 0)) == date(2026, 7, 1)


@pytest.mark.parametrize(
    "epoch_ms",
    [1782878400000, 1783310400000, 1772946000000, 1793599200000],
)
def test_silver_and_ingestion_agree_on_the_session_date(epoch_ms):
    # A-2 resolves the watermark session date and A-3 resolves trade_date. One rule, two callers:
    # this test fails if either implementation is changed alone.
    assert trade_date_from_epoch_ms(epoch_ms) == epoch_ms_to_session_date(epoch_ms)


def test_trade_date_expression_converts_through_the_exchange_timezone():
    assert EXCHANGE_TZ_NAME == "America/New_York"
    assert f"from_utc_timestamp(source_timestamp, '{EXCHANGE_TZ_NAME}')" in TRADE_DATE_EXPR
    assert TRADE_DATE_EXPR.startswith("CAST(") and TRADE_DATE_EXPR.endswith("AS DATE)")


# ========================================================================== volume


def test_volume_rounds_the_fractional_vendor_value():
    # The verified payload sends 1.46147597081851e+08 - fractional shares.
    assert volume_to_long(1.46147597081851e08) == 146147597


def test_volume_rounds_halves_up_like_sql():
    # Python's round() uses banker's rounding and would answer 2 here, disagreeing with SQL.
    assert volume_to_long(2.5) == 3
    assert volume_to_long(3.5) == 4


def test_volume_passes_through_none():
    assert volume_to_long(None) is None


def test_volume_expression_rounds_rather_than_truncating():
    assert VOLUME_EXPR == "CAST(round(volume) AS BIGINT)"


# ===================================================================== generated SQL


def test_price_source_sql_deduplicates_to_one_row_per_session():
    sql = build_price_source_sql(CATALOG)

    assert "market_intel.bronze.prices_raw" in sql
    assert "ROW_NUMBER() OVER (PARTITION BY ticker, " in sql
    assert "WHERE _rn = 1" in sql
    # The derived key repeats its expression: a window cannot reference an alias from its own
    # SELECT, so PARTITION BY trade_date would not compile.
    assert f"PARTITION BY ticker, {TRADE_DATE_EXPR}" in sql
    assert "ORDER BY ingested_at DESC, t_epoch_ms DESC" in sql


def test_price_source_sql_projects_exactly_the_table_columns():
    sql = build_price_source_sql(CATALOG)

    assert DAILY_PRICE_COLUMNS == ("ticker", "trade_date", "open", "high", "low", "close", "volume", "vwap")
    for column in DAILY_PRICE_COLUMNS:
        assert f"`{column}`" in sql


def test_news_source_sql_deduplicates_on_the_merge_key():
    sql = build_news_source_sql(CATALOG)

    assert "market_intel.bronze.news_raw" in sql
    assert "PARTITION BY article_id, ticker" in sql
    assert "ORDER BY ingested_at DESC" in sql
    assert "WHERE _rn = 1" in sql


def test_news_source_sql_projects_exactly_the_table_columns():
    sql = build_news_source_sql(CATALOG)

    assert NEWS_ARTICLE_COLUMNS == (
        "article_id",
        "ticker",
        "published_at",
        "title",
        "description",
        "publisher",
        "sentiment_label",
        "sentiment_score",
        "sentiment_reasoning",
        "embedding_text",
        "article_url",
    )
    for column in NEWS_ARTICLE_COLUMNS:
        assert f"`{column}`" in sql


def test_news_source_sql_maps_publisher_and_label_from_bronze_columns():
    sql = build_news_source_sql(CATALOG)

    assert "publisher_name AS `publisher`" in sql
    assert "sentiment AS `sentiment_label`" in sql
    assert PUBLISHED_AT_EXPR in sql


def test_published_at_falls_back_to_the_instant_bronze_already_parsed():
    assert PUBLISHED_AT_EXPR == "coalesce(try_to_timestamp(published_utc), source_timestamp)"


@pytest.mark.parametrize(
    "keys",
    [PRICE_MERGE_KEYS, NEWS_MERGE_KEYS],
    ids=["prices", "news"],
)
def test_merge_sql_upserts_on_the_declared_keys(keys):
    sql = merge_sql("market_intel.silver.t", "SELECT 1", keys)

    for key in keys:
        assert f"t.`{key}` = s.`{key}`" in sql
    assert "WHEN MATCHED THEN UPDATE SET *" in sql
    assert "WHEN NOT MATCHED THEN INSERT *" in sql
    # Rule 6: never a blind INSERT, so the retrying daily workflow cannot duplicate rows.
    assert "INSERT INTO" not in sql


def test_merge_keys_are_the_spec_keys():
    assert PRICE_MERGE_KEYS == ("ticker", "trade_date")
    assert NEWS_MERGE_KEYS == ("article_id", "ticker")


def test_latest_per_key_sql_rejects_an_empty_projection():
    with pytest.raises(ValueError):
        latest_per_key_sql("t", (), ("k",), "k")


def test_merge_sql_rejects_empty_keys():
    with pytest.raises(ValueError):
        merge_sql("t", "SELECT 1", ())


# ============================================================================ DDL


def test_news_articles_enables_change_data_feed_at_creation():
    block = _ddl_block("silver.news_articles")

    # An AI Search index that syncs to zero rows is almost always this property missing at CREATE
    # time; adding it later does not backfill the feed.
    assert "delta.enableChangeDataFeed = true" in block


def test_daily_prices_does_not_need_change_data_feed():
    assert "enableChangeDataFeed" not in _ddl_block("silver.daily_prices")


def test_silver_ddl_declares_every_projected_column():
    prices = _ddl_block("silver.daily_prices")
    news = _ddl_block("silver.news_articles")

    for column in DAILY_PRICE_COLUMNS:
        assert column in prices, f"daily_prices DDL is missing {column}"
    for column in NEWS_ARTICLE_COLUMNS:
        assert column in news, f"news_articles DDL is missing {column}"


def test_silver_ddl_types_match_the_derivations():
    prices = _ddl_block("silver.daily_prices")
    news = _ddl_block("silver.news_articles")

    assert re.search(r"\btrade_date\s+DATE\b", prices)
    assert re.search(r"\bvolume\s+BIGINT\b", prices)  # cast to LONG from the bronze DOUBLE
    assert re.search(r"\bpublished_at\s+TIMESTAMP\b", news)
    assert re.search(r"\bsentiment_score\s+INT\b", news)


# =====================================================================================
# TODO (integration, run in the Databricks workspace - not fakeable locally):
#
# These need real Delta and a real SparkSession, exactly as with A-2.
#
# 1. IDEMPOTENCY (spec A-5 test_idempotency, rule 6). Run silver_prices.main twice and
#    silver_news.main twice over unchanged bronze; assert identical row counts AND identical
#    checksums both times. Row counts alone would miss a MERGE that rewrites rows it should
#    have left alone.
# 2. THE EXPLODE RULE END TO END. Ingest the strict-subset article, run the silver build, and
#    assert silver.news_articles has rows for the insight tickers and NONE for the tickers that
#    appear only in the raw tickers array.
# 3. TRADE_DATE AGREES WITH PYTHON. For every row, assert the SQL trade_date equals
#    trade_date_from_epoch_ms(t_epoch_ms) computed in the driver - one query, and it is the only
#    way to prove from_utc_timestamp and zoneinfo agree, including across a DST boundary. Add a
#    bronze row stamped 05:00Z in winter for that case.
# 4. UNKNOWN-LABEL WARNING. Insert a bronze row with sentiment 'mixed', run the build, assert
#    sentiment_score = 0, sentiment_label = 'mixed' (kept verbatim) and that the driver log
#    contains the unrecognized-label warning with a row count.
# 5. CDF IS LIVE. DESCRIBE DETAIL silver.news_articles and assert delta.enableChangeDataFeed is
#    true on the CREATED table, then read table_changes() after a build to confirm the feed the
#    AI Search index depends on is actually populated.
# 6. VOLUME CAST. Assert silver volume equals round(bronze volume) for every row and that the
#    column is a LONG, not a DOUBLE.
# =====================================================================================
