"""Feature-pipeline tests (spec A-4/A-5), plus the pipeline ledger rows A-4 adds.

WHAT IS PINNED HERE. Every rolling column is checked against a value computed by hand on a
synthetic series with a closed form, not against a re-implementation of the same loop and not
against an approximate smoke check. The series are chosen so the answer can be written down:

- constant log returns: ``close[i] = 100 * 1.01**i`` gives ``log_return == ln(1.01)`` on every row,
  ``momentum_5d == 5*ln(1.01)``, ``return_5d == 1.01**5 - 1`` and ``realized_vol_20d == 0``.
- alternating log returns ``+r, -r``: a 20-row window has mean 0 and ``stddev_samp == r*sqrt(20/19)``.
- volumes ``[100]*19 + [200]``: mean 105, ``stddev_samp == sqrt(500)``, so the z-score at row 19 is
  ``95/sqrt(500)``.

THE WARM-UP BOUNDARIES ARE PART OF THE CONTRACT, so they are asserted explicitly: the first
non-null ``momentum_5d`` is row 5 (not row 4, even though a 5-row frame exists there — the frame's
first ``log_return`` is NULL), the first ``realized_vol_20d`` is row 20, and the first
``volume_zscore_20d`` is row 19.

THE CALENDAR TESTS ARE THE ONES THAT MATTER MOST. ``test_weekend_news`` proves Saturday and Sunday
articles land in Monday's ``s_t`` and ``news_count``, and the holiday tests prove the mapping
consults the XNYS calendar rather than weekday arithmetic: Good Friday 2026-04-03 is a Friday, and
an article published that day must land on Monday 2026-04-06. A weekday-arithmetic implementation
passes the weekend test and fails the holiday one, which is exactly why both exist.

The rolling formulas execute as Spark SQL in production. These tests pin the Python reference
implementations and assert the generated SQL is built from the same constants; a row-by-row
comparison of the two needs a SparkSession.

TODO(integration, workspace): run the A-4 build twice against Delta and assert
``silver.daily_features`` row count is unchanged and warm-up NULLs survive the MERGE, that the SQL
and Python columns agree row-by-row for one ticker, and that the ``bronze.ingestion_runs`` row
count grows by exactly one per pipeline task per run.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ingestion import RUNS_COLUMNS, RUNS_SCHEMA_DDL, STATUS_FAILED, STATUS_SUCCEEDED
from src.pipelines import EXCHANGE_TZ_NAME, feature_pipeline, silver_news, silver_prices
from src.pipelines.feature_pipeline import (
    FEATURE_COLUMNS,
    MOMENTUM_WINDOW,
    NEWS_DECAY_DENOMINATOR,
    NEWS_DECAY_WEIGHTS,
    RETURN_LAG,
    SESSION_MAP_VIEW,
    SESSIONS_VIEW,
    VOL_WINDOW,
    VOLUME_WINDOW,
    build_features_sql,
    log_returns,
    momentum_expr,
    news_sentiment_3d,
    news_sentiment_3d_expr,
    next_session_map,
    realized_vol_expr,
    returns_over_lag,
    rolling_stddev_samp,
    rolling_sum,
    rolling_zscore,
    session_for_timestamp,
    session_news_aggregates,
    to_exchange_date,
    trading_sessions,
    volume_zscore_expr,
)

CATALOG = "market_intel"
UTC = timezone.utc


# ------------------------------------------------------------------ synthetic series


def geometric_closes(count: int, step: float = 1.01, start: float = 100.0) -> list[float]:
    """``start * step**i``: a series whose log return is exactly ``ln(step)`` on every row."""
    return [start * step**index for index in range(count)]


def closes_from_log_returns(log_return_series: list[float], start: float = 100.0) -> list[float]:
    """Invert the log-return definition so a chosen return pattern can be asserted exactly."""
    closes = [start]
    for value in log_return_series:
        closes.append(closes[-1] * math.exp(value))
    return closes


# ------------------------------------------------------------------------ log_return


def test_log_return_is_null_on_the_first_row():
    assert log_returns([100.0, 101.0])[0] is None


def test_log_return_exact_values_by_hand():
    # ln(110/100) and ln(99/110), written out rather than recomputed with the same helper.
    values = log_returns([100.0, 110.0, 99.0])

    assert values[1] == pytest.approx(math.log(1.1))
    assert values[2] == pytest.approx(math.log(0.9))


def test_log_return_is_constant_on_a_geometric_series():
    values = log_returns(geometric_closes(25))

    assert values[0] is None
    for value in values[1:]:
        assert value == pytest.approx(math.log(1.01))


def test_log_return_is_null_when_a_close_is_missing_or_non_positive():
    # ln of a non-positive ratio is undefined; the SQL try_divide/ln pair yields NULL, not a crash.
    assert log_returns([100.0, None, 120.0]) == [None, None, None]
    assert log_returns([0.0, 120.0])[1] is None
    assert log_returns([100.0, -5.0])[1] is None


# ------------------------------------------------------------------------- return_5d


def test_return_5d_is_null_until_five_rows_of_history():
    values = returns_over_lag(geometric_closes(10))

    assert values[:RETURN_LAG] == [None] * RETURN_LAG
    assert values[RETURN_LAG] is not None


def test_return_5d_exact_value_on_a_geometric_series():
    # close[5]/close[0] - 1 = 1.01**5 - 1 = 0.0510100501...
    values = returns_over_lag(geometric_closes(12))

    assert values[5] == pytest.approx(1.01**5 - 1)
    assert values[11] == pytest.approx(1.01**5 - 1)
    assert values[5] == pytest.approx(0.0510100501, abs=1e-10)


def test_return_5d_uses_the_close_five_rows_back_not_a_calendar_offset():
    closes = [100.0, 1.0, 1.0, 1.0, 1.0, 200.0]

    assert returns_over_lag(closes)[5] == pytest.approx(1.0)


# ----------------------------------------------------------------------- momentum_5d


def test_momentum_5d_exact_value_on_a_geometric_series():
    values = rolling_sum(log_returns(geometric_closes(12)))

    assert values[5] == pytest.approx(5 * math.log(1.01))
    assert values[11] == pytest.approx(5 * math.log(1.01))


def test_momentum_5d_first_non_null_is_row_five_not_row_four():
    """Row 4 has a full 5-row frame, but its first ``log_return`` is NULL.

    Without the count guard Spark would sum the 4 available returns and label the result 5-day
    momentum. This is the assertion that catches that.
    """
    values = rolling_sum(log_returns(geometric_closes(8)))

    assert values[:MOMENTUM_WINDOW] == [None] * MOMENTUM_WINDOW
    assert values[MOMENTUM_WINDOW] is not None


def test_momentum_5d_equals_the_log_of_the_five_row_price_ratio():
    # A sum of log returns telescopes: sum(ln r_i) == ln(close[i] / close[i-5]).
    closes = [10.0, 12.0, 11.0, 15.0, 14.0, 20.0]
    values = rolling_sum(log_returns(closes))

    assert values[5] == pytest.approx(math.log(20.0 / 10.0))


def test_momentum_5d_is_null_when_a_gap_sits_inside_the_frame():
    """One missing close invalidates TWO log returns — the one into the gap and the one out of it.

    So a gap at row 2 keeps momentum NULL until the frame clears row 3, which is row 8.
    """
    closes = [100.0, 101.0, None, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    returns = log_returns(closes)
    values = rolling_sum(returns)

    assert returns[2] is None and returns[3] is None
    assert values[7] is None
    assert values[8] is not None


# ------------------------------------------------------------------ realized_vol_20d


def test_realized_vol_20d_first_non_null_is_row_twenty():
    values = rolling_stddev_samp(log_returns(geometric_closes(30)))

    assert values[:VOL_WINDOW] == [None] * VOL_WINDOW
    assert values[VOL_WINDOW] is not None


def test_realized_vol_20d_is_zero_on_constant_returns():
    # Zero, not NULL: the statistic is defined, and it is exactly zero.
    values = rolling_stddev_samp(log_returns(geometric_closes(30)))

    assert values[VOL_WINDOW] == pytest.approx(0.0, abs=1e-15)


def test_realized_vol_20d_exact_value_on_alternating_returns():
    """Alternating +r/-r: the 20-row window has mean 0, so stddev_samp == r*sqrt(20/19)."""
    r = 0.02
    pattern = [r if index % 2 == 0 else -r for index in range(30)]
    closes = closes_from_log_returns(pattern)

    values = rolling_stddev_samp(log_returns(closes))
    expected = r * math.sqrt(VOL_WINDOW / (VOL_WINDOW - 1))

    assert values[VOL_WINDOW] == pytest.approx(expected)
    assert values[VOL_WINDOW] == pytest.approx(0.020519567041703, abs=1e-12)


def test_realized_vol_20d_uses_the_sample_denominator():
    """stddev_samp (n-1), per spec — not the population form.

    Two returns of +r and -r have sample deviation r*sqrt(2) and population deviation r, so a
    2-row window separates the two definitions unambiguously.
    """
    r = 0.05
    values = rolling_stddev_samp(log_returns(closes_from_log_returns([r, -r])), window=2)

    assert values[2] == pytest.approx(r * math.sqrt(2))


# ---------------------------------------------------------------- volume_zscore_20d


def test_volume_zscore_20d_exact_value_by_hand():
    """volumes = nineteen 100s then a 200.

    mean = (19*100 + 200)/20 = 105. sum of squared deviations = 19*25 + 95**2 = 9500, so
    stddev_samp = sqrt(9500/19) = sqrt(500), and z = (200-105)/sqrt(500).
    """
    volumes = [100.0] * 19 + [200.0]
    values = rolling_zscore(volumes)

    assert values[19] == pytest.approx(95 / math.sqrt(500))
    assert values[19] == pytest.approx(4.2485291572, abs=1e-9)


def test_volume_zscore_20d_first_non_null_is_row_nineteen():
    """One row earlier than realized_vol_20d: volume has no undefined first row."""
    volumes = [float(100 + index) for index in range(25)]
    values = rolling_zscore(volumes)

    assert values[: VOLUME_WINDOW - 1] == [None] * (VOLUME_WINDOW - 1)
    assert values[VOLUME_WINDOW - 1] is not None


def test_volume_zscore_20d_is_null_on_a_constant_window():
    # Zero deviation: try_divide yields NULL rather than an infinity or a divide-by-zero error.
    assert rolling_zscore([1000.0] * 25)[24] is None


def test_volume_zscore_20d_is_zero_when_volume_sits_at_the_window_mean():
    # 1..19 plus a 10: the mean is (190+10)/20 = 10, which is the row's own volume, so z is 0.
    volumes = [float(index) for index in range(1, 20)] + [10.0]
    values = rolling_zscore(volumes)

    assert values[19] == pytest.approx(0.0)


# ---------------------------------------------------------------- news_sentiment_3d


def test_news_sentiment_3d_weights_and_denominator_match_the_spec():
    assert NEWS_DECAY_WEIGHTS == (1.0, 0.5, 0.25)
    assert NEWS_DECAY_DENOMINATOR == 1.75


def test_news_sentiment_3d_exact_values_after_a_single_positive_session():
    # s = [1, 0, 0, 0]: the 1 decays through the window as 1/1.75, 0.5/1.75, 0.25/1.75, then 0.
    values = news_sentiment_3d([1.0, 0.0, 0.0, 0.0])

    assert values[0] == pytest.approx(1 / 1.75)
    assert values[1] == pytest.approx(0.5 / 1.75)
    assert values[2] == pytest.approx(0.25 / 1.75)
    assert values[3] == pytest.approx(0.0)


def test_news_sentiment_3d_is_one_when_three_positive_sessions_fill_the_window():
    # (1.0 + 0.5 + 0.25)/1.75 == 1 exactly: the denominator normalizes a saturated window.
    values = news_sentiment_3d([1.0, 1.0, 1.0])

    assert values[2] == pytest.approx(1.0)
    assert values[0] == pytest.approx(1 / 1.75)
    assert values[1] == pytest.approx(1.5 / 1.75)


def test_news_sentiment_3d_treats_missing_lags_as_zero_not_null():
    """The first two rows have no lags, and the denominator is fixed at 1.75 regardless.

    No news is a real zero, unlike a rolling statistic that is undefined during warm-up, so this
    column is never NULL.
    """
    values = news_sentiment_3d([-1.0, 0.0])

    assert values[0] == pytest.approx(-1 / 1.75)
    assert values[1] == pytest.approx(-0.5 / 1.75)


def test_news_sentiment_3d_handles_negative_sessions():
    values = news_sentiment_3d([-1.0, -1.0, -1.0])

    assert values[2] == pytest.approx(-1.0)


# ----------------------------------------------------------- calendar: session mapping


def test_next_session_map_maps_a_session_to_itself():
    sessions = [date(2026, 8, 10), date(2026, 8, 11)]
    mapping = next_session_map(sessions, date(2026, 8, 10), date(2026, 8, 11))

    assert mapping[date(2026, 8, 10)] == date(2026, 8, 10)
    assert mapping[date(2026, 8, 11)] == date(2026, 8, 11)


def test_next_session_map_rolls_a_closed_day_forward_never_backward():
    sessions = [date(2026, 8, 7), date(2026, 8, 10)]
    mapping = next_session_map(sessions, date(2026, 8, 7), date(2026, 8, 10))

    assert mapping[date(2026, 8, 8)] == date(2026, 8, 10)
    assert mapping[date(2026, 8, 9)] == date(2026, 8, 10)


def test_next_session_map_omits_dates_past_the_last_session():
    """An article that rolls past the calendar has no session to belong to yet, so it is absent
    rather than assigned to a guessed future date."""
    sessions = [date(2026, 8, 10)]
    mapping = next_session_map(sessions, date(2026, 8, 10), date(2026, 8, 12))

    assert mapping == {date(2026, 8, 10): date(2026, 8, 10)}


def test_next_session_map_covers_every_date_in_the_range():
    sessions = trading_sessions(date(2026, 1, 2), date(2026, 3, 31))
    mapping = next_session_map(sessions, date(2026, 1, 2), date(2026, 3, 31))
    span = (date(2026, 3, 31) - date(2026, 1, 2)).days + 1

    assert len(mapping) == span
    assert all(value in set(sessions) for value in mapping.values())


def test_trading_sessions_excludes_weekends_and_holidays():
    sessions = set(trading_sessions(date(2026, 1, 1), date(2026, 12, 31)))

    assert date(2026, 1, 1) not in sessions  # New Year's Day
    assert date(2026, 8, 8) not in sessions  # Saturday
    assert date(2026, 8, 9) not in sessions  # Sunday
    assert date(2026, 8, 10) in sessions  # Monday


# ------------------------------------------------- calendar: weekend and holiday rolls


def test_to_exchange_date_converts_utc_to_the_new_york_date():
    # 2026-08-10T02:15Z is still Sunday evening in New York (22:15 on the 9th).
    assert to_exchange_date(datetime(2026, 8, 10, 2, 15, tzinfo=UTC)) == date(2026, 8, 9)
    assert to_exchange_date(datetime(2026, 8, 8, 20, 5, tzinfo=UTC)) == date(2026, 8, 8)


def test_to_exchange_date_treats_naive_input_as_utc():
    assert to_exchange_date(datetime(2026, 8, 10, 2, 15)) == date(2026, 8, 9)


def test_weekend_news():
    """Saturday and Sunday articles land in MONDAY's ``s_t`` and ``news_count`` (spec A-5).

    Three articles: Saturday +1, Sunday -1, Monday +1. All three belong to Monday's session, so
    ``news_count`` is 3 and ``s_t`` is their mean, 1/3. Friday's session gets nothing.
    """
    sessions = trading_sessions(date(2026, 8, 3), date(2026, 8, 14))
    session_map = next_session_map(sessions, date(2026, 8, 3), date(2026, 8, 14))
    monday = date(2026, 8, 10)

    saturday = datetime(2026, 8, 8, 20, 5, tzinfo=UTC)  # 16:05 Saturday in New York
    sunday = datetime(2026, 8, 9, 13, 45, tzinfo=UTC)  # 09:45 Sunday in New York
    monday_intraday = datetime(2026, 8, 10, 17, 30, tzinfo=UTC)  # 13:30 Monday, market open
    articles = [
        {"ticker": "NVDA", "published_at": saturday, "sentiment_score": 1},
        {"ticker": "NVDA", "published_at": sunday, "sentiment_score": -1},
        {"ticker": "NVDA", "published_at": monday_intraday, "sentiment_score": 1},
    ]
    aggregates = session_news_aggregates(articles, session_map)

    assert list(aggregates) == [("NVDA", monday)]
    s_t, news_count = aggregates[("NVDA", monday)]
    assert news_count == 3
    assert s_t == pytest.approx(1 / 3)
    assert ("NVDA", date(2026, 8, 7)) not in aggregates  # the preceding Friday stays empty


def test_weekend_news_from_verified_payload_timestamps(news_results: list[dict]):
    """The same rule applied to the live-verified fixture timestamps.

    All three fixture articles are stamped Saturday 16:05, Sunday 09:45 and Sunday 22:15 New York
    time, so every one of them belongs to Monday 2026-08-10.
    """
    from src.ingestion.ingest_news import parse_published_utc
    from src.pipelines.silver_news import sentiment_score

    sessions = trading_sessions(date(2026, 8, 3), date(2026, 8, 14))
    session_map = next_session_map(sessions, date(2026, 8, 3), date(2026, 8, 14))

    articles = [
        {
            "ticker": insight["ticker"],
            "published_at": parse_published_utc(article["published_utc"]),
            "sentiment_score": sentiment_score(insight["sentiment"]),
        }
        for article in news_results
        for insight in article["insights"]
    ]
    aggregates = session_news_aggregates(articles, session_map)

    assert {session for _, session in aggregates} == {date(2026, 8, 10)}
    assert aggregates[("MSFT", date(2026, 8, 10))] == (1.0, 1)
    assert aggregates[("SNDK", date(2026, 8, 10))] == (1.0, 1)
    # NVDA: one neutral insight and one unrecognized label scored 0 (A-3), both on Monday.
    assert aggregates[("NVDA", date(2026, 8, 10))] == (0.0, 2)


@pytest.mark.parametrize(
    ("holiday", "expected_session", "label"),
    [
        (date(2026, 4, 3), date(2026, 4, 6), "Good Friday"),
        (date(2026, 7, 3), date(2026, 7, 6), "Independence Day observed"),
        (date(2026, 1, 19), date(2026, 1, 20), "Martin Luther King Jr. Day"),
    ],
)
def test_holiday_news_rolls_to_the_next_session_not_the_next_weekday(
    holiday: date,
    expected_session: date,
    label: str,
):
    """A WEEKDAY holiday, which is where weekday arithmetic fails.

    Good Friday 2026-04-03 is a Friday and 2026-07-03 is a Friday: weekday arithmetic assigns an
    article published on either to that same day, because it is a business day. The calendar
    assigns it to the following Monday. MLK Day is a Monday, so its article belongs to Tuesday,
    not to the Monday it was published on and not to the "next weekday" from Friday.
    """
    sessions = trading_sessions(holiday - timedelta(days=10), holiday + timedelta(days=10))
    session_map = next_session_map(
        sessions, holiday - timedelta(days=10), holiday + timedelta(days=10)
    )

    assert holiday not in set(sessions), f"{label} should not be an XNYS session"
    assert holiday.weekday() < 5, f"{label} must be a weekday for this test to mean anything"
    assert session_map[holiday] == expected_session
    assert session_map[holiday] != holiday


def test_thanksgiving_rolls_to_the_half_day_friday_session():
    """The day after Thanksgiving is a shortened session, but it IS a session.

    The rule is "next session", not "next full session", so an article published on Thanksgiving
    belongs to Friday rather than being pushed to the following Monday.
    """
    sessions = trading_sessions(date(2026, 11, 20), date(2026, 12, 4))
    session_map = next_session_map(sessions, date(2026, 11, 20), date(2026, 12, 4))

    assert session_map[date(2026, 11, 26)] == date(2026, 11, 27)


def test_session_for_timestamp_maps_a_saturday_evening_article_to_monday():
    sessions = trading_sessions(date(2026, 8, 3), date(2026, 8, 14))
    session_map = next_session_map(sessions, date(2026, 8, 3), date(2026, 8, 14))

    saturday_evening = datetime(2026, 8, 9, 1, 30, tzinfo=UTC)  # 21:30 Saturday in New York

    assert session_for_timestamp(saturday_evening, session_map) == date(2026, 8, 10)


def test_session_for_timestamp_returns_none_past_the_calendar():
    session_map = {date(2026, 8, 10): date(2026, 8, 10)}

    assert session_for_timestamp(datetime(2026, 8, 20, 12, 0, tzinfo=UTC), session_map) is None


# ------------------------------------------------------------- news aggregation detail


def test_session_news_aggregates_keeps_tickers_separate():
    session_map = {date(2026, 8, 10): date(2026, 8, 10)}
    published = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    articles = [
        {"ticker": "NVDA", "published_at": published, "sentiment_score": 1},
        {"ticker": "MSFT", "published_at": published, "sentiment_score": -1},
    ]

    aggregates = session_news_aggregates(articles, session_map)

    assert aggregates[("NVDA", date(2026, 8, 10))] == (1.0, 1)
    assert aggregates[("MSFT", date(2026, 8, 10))] == (-1.0, 1)


def test_session_news_aggregates_counts_a_null_score_as_zero():
    session_map = {date(2026, 8, 10): date(2026, 8, 10)}
    published = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    articles = [
        {"ticker": "NVDA", "published_at": published, "sentiment_score": None},
        {"ticker": "NVDA", "published_at": published, "sentiment_score": 1},
    ]

    aggregates = session_news_aggregates(articles, session_map)

    assert aggregates[("NVDA", date(2026, 8, 10))] == (0.5, 2)


def test_session_news_aggregates_warns_when_an_article_has_no_session(caplog):
    session_map = {date(2026, 8, 10): date(2026, 8, 10)}
    articles = [
        {"ticker": "NVDA", "published_at": datetime(2026, 9, 1, 14, 0, tzinfo=UTC), "sentiment_score": 1}
    ]

    with caplog.at_level(logging.WARNING, logger="src.pipelines.feature_pipeline"):
        aggregates = session_news_aggregates(articles, session_map)

    assert aggregates == {}
    assert "no session" in caplog.text


# ------------------------------------------------------- generated SQL matches Python


def test_sql_window_frames_are_generated_from_the_window_constants():
    assert f"{MOMENTUM_WINDOW - 1} PRECEDING" in build_features_sql(CATALOG)
    assert f"{VOL_WINDOW - 1} PRECEDING" in build_features_sql(CATALOG)
    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in build_features_sql(CATALOG)
    assert "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW" in build_features_sql(CATALOG)


def test_sql_guards_every_rolling_column_with_a_full_frame_count():
    """The count guard is what makes the Python NULLs and the SQL NULLs agree.

    Spark's aggregates skip NULLs, so without this a partial frame would produce a number.
    """
    assert f"count(log_return) OVER w_momentum = {MOMENTUM_WINDOW}" in momentum_expr()
    assert f"count(log_return) OVER w_vol = {VOL_WINDOW}" in realized_vol_expr()
    assert f"count(volume) OVER w_volume = {VOLUME_WINDOW}" in volume_zscore_expr()


def test_sql_uses_the_sample_standard_deviation_for_both_columns():
    assert "stddev_samp(log_return)" in realized_vol_expr()
    assert "stddev_samp(volume)" in volume_zscore_expr()
    assert "stddev_pop" not in build_features_sql(CATALOG)


def test_sql_divides_with_try_divide_so_a_zero_denominator_yields_null():
    assert "try_divide" in volume_zscore_expr()
    assert "ln(try_divide(" in build_features_sql(CATALOG)


def test_sql_news_decay_expression_is_generated_from_the_weights():
    expression = news_sentiment_3d_expr()

    assert "1.0 * s_t" in expression
    assert "0.5 * coalesce(lag(s_t, 1) OVER w_order, 0.0)" in expression
    assert "0.25 * coalesce(lag(s_t, 2) OVER w_order, 0.0)" in expression
    assert expression.endswith(f"/ {NEWS_DECAY_DENOMINATOR}")


def test_sql_backquotes_the_close_column_everywhere_it_appears():
    """``close`` is quoted so it cannot read as a keyword, and its output alias must be quoted
    too — ``INSERT *`` matches the target column by name."""
    sql = build_features_sql(CATALOG)

    assert "`close` AS `close`" in sql
    assert "lag(`close`, 1)" in sql
    assert re.search(r"[^`.]close[^`]", sql.replace("`close`", "")) is None


def test_sql_partitions_by_ticker_and_orders_by_trade_date():
    sql = build_features_sql(CATALOG)

    assert sql.count("PARTITION BY ticker ORDER BY trade_date") == 5
    assert "ORDER BY trade_date DESC" not in sql


def test_sql_reads_silver_and_never_bronze():
    sql = build_features_sql(CATALOG)

    assert "market_intel.silver.daily_prices" in sql
    assert "market_intel.silver.news_articles" in sql
    assert "bronze" not in sql


def test_sql_restricts_the_grain_to_calendar_sessions():
    sql = build_features_sql(CATALOG)

    assert f"JOIN {SESSIONS_VIEW} s ON p.trade_date = s.session_date" in sql
    assert f"JOIN {SESSION_MAP_VIEW} m ON" in sql


def test_sql_maps_news_through_the_exchange_timezone():
    sql = build_features_sql(CATALOG)

    assert f"from_utc_timestamp(n.published_at, '{EXCHANGE_TZ_NAME}')" in sql
    assert EXCHANGE_TZ_NAME == "America/New_York"


def test_sql_zero_fills_sessions_without_news_instead_of_dropping_them():
    sql = build_features_sql(CATALOG)

    assert "LEFT JOIN" in sql
    assert "coalesce(nw.s_t, 0.0) AS s_t" in sql
    assert "coalesce(nw.news_count, 0) AS news_count" in sql


def test_sql_averages_sentiment_score_per_session():
    sql = build_features_sql(CATALOG)

    assert "avg(coalesce(CAST(n.sentiment_score AS DOUBLE), 0.0)) AS s_t" in sql
    assert "count(*) AS news_count" in sql


def test_sql_counts_a_null_score_as_zero_so_s_t_and_news_count_describe_the_same_articles():
    """Spark's ``avg`` skips NULLs but ``count(*)`` does not, so the mean must coalesce.

    ``session_news_aggregates`` counts a missing score as 0 for the same reason, and A-3 already
    degrades an unscored article to neutral rather than dropping it.
    """
    assert "avg(coalesce(" in build_features_sql(CATALOG)


def test_feature_columns_match_the_delta_ddl_order():
    """The projection order and the table's column order must agree: the MERGE inserts by
    position via ``INSERT *``."""
    block = _daily_features_ddl_block()
    declared = [
        line.strip().split()[0].strip("`")
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    assert tuple(declared) == FEATURE_COLUMNS


def test_daily_features_ddl_declares_the_merge_keys_not_null():
    block = _daily_features_ddl_block()

    assert re.search(r"^\s*ticker\s+STRING NOT NULL", block, re.MULTILINE)
    assert re.search(r"^\s*trade_date\s+DATE\s+NOT NULL", block, re.MULTILINE)
    assert feature_pipeline.MERGE_KEYS == ("ticker", "trade_date")


# ------------------------------------------------------------- ingestion_runs ledger


class _FakeResult:
    def __init__(self, row: dict | None = None, rows: list | None = None):
        self._row = row
        self._rows = rows or []

    def first(self):
        return self._row

    def collect(self):
        return self._rows


class _FakeFrame:
    def __init__(self, spark: "_FakeSpark", rows: list, schema: str):
        self._spark = spark
        self.rows = rows
        self.schema = schema

    def createOrReplaceTempView(self, name: str) -> None:  # noqa: N802 — Spark's API
        self._spark.views[name] = self


class _FakeCatalog:
    def __init__(self, spark: "_FakeSpark"):
        self._spark = spark

    def tableExists(self, fqn: str) -> bool:  # noqa: N802 — Spark's API
        return fqn not in self._spark.missing_tables

    def dropTempView(self, name: str) -> None:  # noqa: N802 — Spark's API
        self._spark.views.pop(name, None)


class _FakeConf:
    def __init__(self):
        self.values: dict[str, str] = {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


class _FakeSpark:
    """Enough of a SparkSession to run a pipeline's ``main`` and inspect what it wrote.

    This is not a substitute for running against Delta — the SQL is never parsed — but the ledger
    contract is Python control flow, not SQL, and that is what these tests pin: one row per call,
    on success and on failure, with the right task name.
    """

    def __init__(self, *, row_count: int = 7, fail_on: str | None = None):
        self.catalog = _FakeCatalog(self)
        self.conf = _FakeConf()
        self.views: dict[str, _FakeFrame] = {}
        self.missing_tables: set[str] = set()
        self.statements: list[str] = []
        self.frames: list[_FakeFrame] = []
        self._row_count = row_count
        self._fail_on = fail_on

    def sql(self, text: str, args: dict | None = None) -> _FakeResult:
        self.statements.append(text)
        if self._fail_on and self._fail_on in text:
            raise RuntimeError("boom: simulated Spark failure")
        if "min(trade_date)" in text:
            return _FakeResult({"first_date": date(2026, 8, 3), "last_date": date(2026, 8, 14)})
        if "count(*) AS n FROM" in text:
            return _FakeResult({"n": self._row_count})
        return _FakeResult()

    def createDataFrame(self, rows, schema: str) -> _FakeFrame:  # noqa: N802 — Spark's API
        frame = _FakeFrame(self, list(rows), schema)
        self.frames.append(frame)
        return frame

    def ledger_rows(self) -> list[dict]:
        return [
            dict(zip(RUNS_COLUMNS, row))
            for frame in self.frames
            if frame.schema == RUNS_SCHEMA_DDL
            for row in frame.rows
        ]


@pytest.mark.parametrize(
    ("module", "expected_task"),
    [
        (silver_prices, "build_silver_prices"),
        (silver_news, "build_silver_news"),
        (feature_pipeline, "build_features"),
    ],
)
def test_every_pipeline_task_writes_one_ledger_row_on_success(module, expected_task: str):
    spark = _FakeSpark(row_count=7)

    summary = module.main(spark, {"catalog": CATALOG})
    rows = spark.ledger_rows()

    assert module.TASK_NAME == expected_task
    assert len(rows) == 1
    assert rows[0]["task"] == expected_task
    assert rows[0]["status"] == STATUS_SUCCEEDED
    assert rows[0]["rows_written"] == 7
    assert rows[0]["error"] is None
    assert summary["rows_merged"] == 7
    assert summary["run_id"] == rows[0]["run_id"]


@pytest.mark.parametrize(
    "module",
    [silver_prices, silver_news, feature_pipeline],
)
def test_every_pipeline_task_writes_a_failed_ledger_row_and_reraises(module):
    # Fail inside the build, before the ledger's own write, so the ledger row still lands.
    spark = _FakeSpark(fail_on="count(*) AS n FROM")

    with pytest.raises(RuntimeError, match="simulated Spark failure"):
        module.main(spark, {"catalog": CATALOG})

    rows = spark.ledger_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_FAILED
    assert rows[0]["error"].startswith("RuntimeError: boom")
    assert rows[0]["finished_at"] is not None


@pytest.mark.parametrize(
    "module",
    [silver_prices, silver_news, feature_pipeline],
)
def test_every_pipeline_task_stamps_a_utc_window_on_its_ledger_row(module):
    spark = _FakeSpark()

    module.main(spark, {"catalog": CATALOG})
    row = spark.ledger_rows()[0]

    assert row["started_at"].tzinfo is not None
    assert row["finished_at"] >= row["started_at"]


def test_ledger_task_names_are_documented_in_the_ddl():
    """The ``task`` column's comment is the only place an operator learns the vocabulary."""
    ddl = _read_ddl()

    for task in ("ingest_prices", "ingest_news", "build_silver_prices", "build_silver_news", "build_features"):
        assert task in ddl


def test_feature_pipeline_pins_the_session_timezone_to_utc():
    spark = _FakeSpark()

    feature_pipeline.main(spark, {"catalog": CATALOG})

    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"


def test_feature_pipeline_registers_both_calendar_views():
    spark = _FakeSpark()

    feature_pipeline.main(spark, {"catalog": CATALOG})

    assert SESSIONS_VIEW in spark.views
    assert SESSION_MAP_VIEW in spark.views
    # 2026-08-03..2026-08-14 is ten sessions and twelve calendar dates.
    assert len(spark.views[SESSIONS_VIEW].rows) == 10
    assert len(spark.views[SESSION_MAP_VIEW].rows) == 12


def test_feature_pipeline_skips_the_build_when_prices_are_empty():
    spark = _FakeSpark()
    spark.sql = _empty_prices_sql(spark)  # type: ignore[method-assign]

    summary = feature_pipeline.main(spark, {"catalog": CATALOG})

    target = f"MERGE INTO {CATALOG}.silver.daily_features"

    assert summary["rows_merged"] == 0
    assert spark.ledger_rows()[0]["status"] == STATUS_SUCCEEDED
    assert not any(target in statement for statement in spark.statements)


def test_feature_pipeline_fails_with_an_actionable_message_when_a_table_is_missing():
    spark = _FakeSpark()
    spark.missing_tables.add(f"{CATALOG}.silver.daily_features")

    with pytest.raises(RuntimeError, match="daily_features"):
        feature_pipeline.main(spark, {"catalog": CATALOG})

    assert spark.ledger_rows()[0]["status"] == STATUS_FAILED


def _empty_prices_sql(spark: _FakeSpark):
    original = spark.sql

    def sql(text: str, args: dict | None = None):
        if "min(trade_date)" in text:
            spark.statements.append(text)
            return _FakeResult({"first_date": None, "last_date": None})
        return original(text, args)

    return sql


def _read_ddl() -> str:
    path = Path(__file__).resolve().parents[1] / "setup" / "create_delta_tables.sql"
    return path.read_text(encoding="utf-8")


def _daily_features_ddl_block() -> str:
    """The column declarations of ``silver.daily_features``, without the surrounding statement."""
    ddl = _read_ddl()
    return ddl.split("market_intel.silver.daily_features (", 1)[1].split(")\nUSING DELTA", 1)[0]
