"""Market Research page (spec A2, C-5).

Select a ticker; read, in this order, what the forecast says, how wide it is, and why any of it
should be believed. Every number is READ from Gold. This page computes no statistics — if a number
is not in ``gold.regime_states``, ``gold.forecast_runs`` or ``gold.backtest_summary``, it is not
shown.

THE ORDER IS THE ARGUMENT. The headline card answers the question in one assembled sentence, the
range card says how uncertain that answer is, and the trust block says what the answer has been
measured against — then the price chart, the regime parameters and the article list, which are the
evidence a reader goes looking for once the first three have earned the attention. The earlier
layout led with a year of price history, which is the one thing on the page the system did not
produce.

NO MODEL NAME REACHES THE TRUST BLOCK. "news_markov vs markov vs gbm" is an implementation detail
of the evaluation; a reader of a single forecast needs the sample size, the outcome and the
reliability facts, and the arm names live in the Model Evaluation page and the README. The lines
themselves are built in :mod:`app.pages.model_evaluation`, which already owns that table and the
statistics over it.

TWO THINGS ARE STATED EVEN WHEN THEY ARE INCONVENIENT:

- The news-decay assumption, next to the forecast rather than in a footer (spec A2). The model
  conditions on today's news and decays it; it does not know tomorrow's.
- "No relevant news" as distinct from "neutral news". A ticker with an empty window and a ticker
  the press is indifferent about produce the same sentiment number and mean completely different
  things, and the tone line says which one this is.

A GBM DAY HAS NO REGIME. When the ladder falls back to Model A there are no regimes, the daily task
writes no ``regime_states`` row, and the badge and the verdict both say so instead of rendering
zeros. The trust block adds its own line when the day's forecast came from a fallback rung.

The pure functions at the top are what the tests exercise: they turn a Gold row into the sentence
the card shows, with no Streamlit and no warehouse anywhere near them.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402 — the path has to be set before the app imports

from app.common import (  # noqa: E402
    CACHE_TTL_SECONDS,
    LAKEBASE_HINT,
    MISSING,
    WAREHOUSE_HINT,
    as_datetime,
    as_float,
    age_phrase,
    catalog,
    decay_disclosure,
    fixed,
    horizon_days,
    money,
    number,
    pct,
    seed_tickers,
)
from app.pages import model_evaluation  # noqa: E402
from src.database import delta  # noqa: E402

TITLE = "Market Research"

#: The trust block's heading, in the words a reader would use for the question it answers.
TRUST_TITLE = "How much to trust this"

#: How much price history the chart shows. A year of sessions is enough to see the regime the
#: model is talking about without turning the chart into a decade of scenery.
CHART_DAYS = 365

#: How many headlines the list shows. The window in silver.news_recent is 90 days; this is the
#: page's own cap on how much of it lands on screen at once.
NEWS_LIMIT = 25

PRICES_TABLE = "silver.daily_prices"
REGIME_TABLE = "gold.regime_states"
FORECAST_TABLE = "gold.forecast_runs"
NEWS_TABLE = "silver.news_recent"

#: Vendor labels, mapped to the marker the list shows. An unrecognized label is displayed
#: verbatim with a neutral marker rather than silently recoded (A-3 keeps the raw label for
#: exactly this reason).
SENTIMENT_MARKERS = {"positive": "▲", "negative": "▼", "neutral": "•"}


# ------------------------------------------------------------------ pure functions


def regime_headline(row: Mapping[str, Any] | None) -> str:
    """"High volatility — 73%", or an honest sentence when there is no regime row."""
    if not row:
        return "No regime estimate for this ticker yet."

    low = row.get("prob_low_vol")
    high = row.get("prob_high_vol")
    if low is None and high is None:
        return "No regime estimate: the model that produced this day has no regimes."

    low_value = float(low or 0.0)
    high_value = float(high or 0.0)
    if high_value >= low_value:
        return f"High volatility — {pct(high_value, 0)}"
    return f"Low volatility — {pct(low_value, 0)}"


def regime_detail(row: Mapping[str, Any] | None) -> str:
    """The parameters behind the headline, as one line. Daily means, annualized nothing."""
    if not row:
        return ""
    return (
        f"Calm regime: {pct(row.get('low_vol_mean'), 2, signed=True)} mean daily return, "
        f"{pct(row.get('low_vol_sigma'), 2)} daily volatility. "
        f"Turbulent regime: {pct(row.get('high_vol_mean'), 2, signed=True)} mean, "
        f"{pct(row.get('high_vol_sigma'), 2)} volatility."
    )


def news_tone(rows: Sequence[Mapping[str, Any]]) -> str:
    """The news_count context line (spec A2): silence and indifference are different states."""
    if not rows:
        return "No news in the window. The forecast's news signal is therefore zero — that is an absence of news, not a negative view."

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("sentiment_label") or "unlabelled").strip().lower()
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    breakdown = ", ".join(f"{count} {label}" for label, count in ranked)
    return f"{total} article{'s' if total != 1 else ''} in the window: {breakdown}."


def chart_data(rows: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """Price rows as columns for ``st.line_chart``.

    A dict rather than a DataFrame so this module never imports pandas: Streamlit converts it, and
    the app's requirements stay the short list of things the app genuinely depends on.
    """
    return {
        "trade_date": [row.get("trade_date") for row in rows],
        "close": [None if row.get("close") is None else float(row["close"]) for row in rows],
    }


def forecast_caption(row: Mapping[str, Any] | None) -> str:
    """Provenance for the forecast card: which model, which day, how many paths."""
    if not row:
        return "No forecast has been computed for this ticker yet."
    return (
        f"Model {row.get('model_used', MISSING)} · as of {row.get('as_of_date', MISSING)} · "
        f"{number(row.get('n_paths'))} simulated paths · "
        f"{number(row.get('horizon_days'))}-day horizon"
    )


def sentiment_marker(label: Any) -> str:
    return SENTIMENT_MARKERS.get(str(label or "").strip().lower(), "•")


# ------------------------------------------------------------------ the headline
#
# WHAT A READER SEES FIRST IS ONE SENTENCE, AND IT IS ASSEMBLED, NOT WRITTEN. The verdict is a
# template filled from two bands — the direction probability and the regime — so it cannot drift
# into a recommendation, cannot hedge differently on two days with the same numbers, and cannot say
# anything the tables do not support. No free text reaches this line and no model generates it.


#: The bands the direction phrase is cut at. A 54% chance of a positive return is not "positive":
#: it is barely off a coin flip, and the first line a user reads has to say that rather than
#: rounding a near-tie into a direction they will act on.
COIN_FLIP_FLOOR = 0.45
COIN_FLIP_CEILING = 0.55

LEANING_POSITIVE = "leaning positive"
LEANING_NEGATIVE = "leaning negative"
COIN_FLIP = "close to a coin flip"

CALM = "calm"
TURBULENT = "turbulent"
UNKNOWN = "unknown"

#: Badge text per tone. The tone drives the colour and nothing else — a turbulent market is styled
#: amber because it is worth noticing, not because the wording changes.
REGIME_BADGES = {
    CALM: "Calm market",
    TURBULENT: "Turbulent market",
    UNKNOWN: "Regime unknown",
}


def direction_phrase(prob: Any) -> str | None:
    """Which band the direction probability falls in, or ``None`` when there is no probability."""
    value = as_float(prob)
    if value is None:
        return None
    if value < COIN_FLIP_FLOOR:
        return LEANING_NEGATIVE
    return COIN_FLIP if value <= COIN_FLIP_CEILING else LEANING_POSITIVE


def regime_badge(row: Mapping[str, Any] | None) -> tuple[str, str]:
    """``("Calm market · 98.3%", "calm")`` — the badge text and the tone that colours it.

    Works on a ``gold.regime_states`` row or a ``gold.forecast_runs`` row, since both carry the two
    filtered probabilities. A gbm day has neither and gets the honest badge rather than zeros.
    """
    low = as_float((row or {}).get("prob_low_vol"))
    high = as_float((row or {}).get("prob_high_vol"))
    if low is None and high is None:
        return REGIME_BADGES[UNKNOWN], UNKNOWN

    low_value = low or 0.0
    high_value = high or 0.0
    tone = TURBULENT if high_value >= low_value else CALM
    dominant = high_value if tone == TURBULENT else low_value
    return f"{REGIME_BADGES[tone]} · {pct(dominant)}", tone


def direction_label(forecast: Mapping[str, Any] | None = None) -> str:
    """The lead metric's label. The horizon comes from the row or config, never from prose."""
    return f"Positive {number(forecast_horizon(forecast))}-day return"


def forecast_horizon(forecast: Mapping[str, Any] | None = None) -> int:
    """The horizon this forecast actually used, falling back to the configured one."""
    stored = as_float((forecast or {}).get("horizon_days"))
    return int(stored) if stored else horizon_days()


def headline_verdict(
    forecast: Mapping[str, Any] | None,
    regime: Mapping[str, Any] | None = None,
) -> str:
    """The one assembled sentence. Bands in, plain English out, no advice verb anywhere."""
    if not forecast:
        return "No forecast has been computed for this ticker yet."

    phrase = direction_phrase(forecast.get("prob_positive"))
    if phrase is None:
        return "This forecast carries no direction probability, so there is nothing to summarize."

    days = number(forecast_horizon(forecast))
    _, tone = regime_badge(regime if regime else forecast)
    if tone == UNKNOWN:
        return (
            f"Over the next {days} trading days the distribution is {phrase}, in a market whose "
            "regime the model could not estimate for this day."
        )
    return (
        f"Over the next {days} trading days the distribution is {phrase}, in a market the model "
        f"currently reads as {tone}."
    )


# ------------------------------------------------------------------ the range


#: The three percentiles the card shows, as (label, price column, return column). p10 and p90 are
#: the ends of the 80% interval the backtest scores coverage against, so they are the two the
#: trust block's coverage line is about — the same interval, named the same way in both places.
RANGE_ROWS = (
    ("Low · 10th percentile", "price_p10", "return_p10"),
    ("Median", "price_p50", "return_p50"),
    ("High · 90th percentile", "price_p90", "return_p90"),
)


def range_rows(forecast: Mapping[str, Any] | None) -> list[dict]:
    """Each percentile as a price AND a return, in one row rather than two separate blocks.

    A price alone makes a reader do arithmetic against today's close; a percentage alone hides what
    the position is actually worth. Both, side by side, is the whole point of the card.
    """
    if not forecast:
        return []
    return [
        {
            "label": label,
            "price": money(forecast.get(price_key)),
            "return": pct(forecast.get(return_key), signed=True),
        }
        for label, price_key, return_key in RANGE_ROWS
    ]


def band_offset(forecast: Mapping[str, Any] | None) -> float | None:
    """Where the median sits between the two ends, 0–100, for the visual band's marker.

    ``None`` when the band would be degenerate (missing ends, or p90 not above p10), which is the
    signal to draw the numbers without the bar rather than a bar with the marker pinned at zero.
    """
    low = as_float((forecast or {}).get("price_p10"))
    mid = as_float((forecast or {}).get("price_p50"))
    high = as_float((forecast or {}).get("price_p90"))
    if low is None or mid is None or high is None or high <= low:
        return None
    return max(0.0, min(100.0, (mid - low) / (high - low) * 100.0))


def loss_line(forecast: Mapping[str, Any] | None) -> str:
    """The downside in the plainest words available. Analysis, not a recommendation."""
    probability = as_float((forecast or {}).get("prob_loss_gt_5pct"))
    if probability is None:
        return "Chance of losing more than 5%: not recorded for this forecast."
    return f"Chance of losing more than 5%: {pct(probability)}."


def distribution_caption(forecast: Mapping[str, Any] | None) -> str:
    """What the three numbers ARE. A percentile without its sample size is not interpretable."""
    if not forecast:
        return ""
    paths = forecast.get("n_paths")
    counted = "an unrecorded number of" if paths is None else number(paths)
    return (
        f"Distribution across {counted} simulated paths at a "
        f"{number(forecast_horizon(forecast))}-trading-day horizon."
    )


# ------------------------------------------------------------------ reads


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def price_history(ticker: str, days: int = CHART_DAYS) -> list[dict]:
    start = date.today() - timedelta(days=days)
    return delta.query(
        f"SELECT trade_date, close FROM {delta.qualified(catalog(), PRICES_TABLE)} "
        "WHERE ticker = :ticker AND trade_date >= :start ORDER BY trade_date",
        {"ticker": ticker, "start": start},
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def current_regime(ticker: str) -> dict | None:
    return delta.query_one(
        f"SELECT * FROM {delta.qualified(catalog(), REGIME_TABLE)} "
        "WHERE ticker = :ticker ORDER BY as_of_date DESC LIMIT 1",
        {"ticker": ticker},
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def latest_forecast(ticker: str) -> dict | None:
    return delta.query_one(
        f"SELECT * FROM {delta.qualified(catalog(), FORECAST_TABLE)} "
        "WHERE ticker = :ticker ORDER BY as_of_date DESC, generated_at DESC LIMIT 1",
        {"ticker": ticker},
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def recent_news(ticker: str, limit: int = NEWS_LIMIT) -> list[dict]:
    # LIMIT is composed rather than bound: Databricks wants a constant there, and a parameter
    # marker is not one. int() is what makes composing it safe, and the ticker — the value that
    # actually comes from outside — still travels as a parameter.
    return delta.query(
        "SELECT title, publisher, sentiment_label, published_at, article_url "
        f"FROM {delta.qualified(catalog(), NEWS_TABLE)} "
        f"WHERE ticker = :ticker ORDER BY published_at DESC LIMIT {int(limit)}",
        {"ticker": ticker},
    )


def watchlist_tickers() -> list[str]:
    """The Lakebase watchlist, or an empty list when Lakebase is unreachable.

    Swallowed deliberately: the watchlist widens the ticker selector, and losing Postgres should
    cost the user AMD in a dropdown, not the whole page. The import is local because it pulls in
    psycopg, and nothing outside the app container may do that.
    """
    try:
        from src.database import lakebase

        return lakebase.get_watchlist()
    except Exception as exc:  # noqa: BLE001 — reported to the user, never raised on the read path
        st.session_state["lakebase_error"] = f"{type(exc).__name__}: {exc}"
        return []


def ticker_options() -> list[str]:
    """Seed universe plus whatever is on the watchlist, deduplicated (spec C-5)."""
    return sorted(set(seed_tickers()) | set(watchlist_tickers()))


# ------------------------------------------------------------------ rendering


def render() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":bar_chart:", layout="wide")
    st.title(TITLE)

    options = ticker_options()
    if not options:
        st.error("No tickers configured. Check `tickers.seed` in `config/config.yaml`.")
        return

    ticker = st.selectbox("Ticker", options, index=0)
    if st.session_state.get("lakebase_error"):
        st.caption(f"{LAKEBASE_HINT} ({st.session_state['lakebase_error']})")

    try:
        prices = price_history(ticker)
        regime = current_regime(ticker)
        forecast = latest_forecast(ticker)
        news = recent_news(ticker)
    except Exception as exc:  # noqa: BLE001 — the diagnosis is known; show it, not a traceback
        st.error(WAREHOUSE_HINT)
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    # TOP TO BOTTOM IS THE ARGUMENT: what the forecast says, how wide it is, and why any of it
    # should be believed. The price history and the article list are evidence a reader goes looking
    # for afterwards, so they sit below the three cards rather than competing with them.
    _render_headline(forecast, regime)
    _render_range(forecast)
    _render_trust(forecast)
    _render_price_chart(ticker, prices)
    _render_regime_detail(regime)
    _render_news(news)


def _render_headline(
    forecast: Mapping[str, Any] | None,
    regime: Mapping[str, Any] | None,
) -> None:
    """Direction, regime, and the assembled sentence — the whole answer, above the fold."""
    if not forecast:
        st.warning(headline_verdict(None))
        st.caption(forecast_caption(None))
        return

    badge, tone = regime_badge(regime if regime else forecast)
    direction, market = st.columns(2)
    direction.metric(direction_label(forecast), pct(forecast.get("prob_positive")))
    market.metric("Market regime", badge)

    # Amber for turbulent. The wording is identical either way — the colour is the only thing the
    # regime changes, because a sentence that gets more alarming is a sentence with an opinion.
    banner = st.warning if tone == TURBULENT else st.info
    banner(headline_verdict(forecast, regime))
    st.caption(forecast_caption(forecast))


def _render_range(forecast: Mapping[str, Any] | None) -> None:
    st.subheader("The range")
    rows = range_rows(forecast)
    if not rows:
        return

    for column, row in zip(st.columns(len(rows)), rows):
        # delta_color="off": a red or green arrow next to a percentile reads as a signal, and the
        # percentile is not one. It is one end of an interval.
        column.metric(row["label"], row["price"], row["return"], delta_color="off")

    offset = band_offset(forecast)
    if offset is not None:
        st.markdown(_band_html(offset), unsafe_allow_html=True)

    st.write(loss_line(forecast))
    st.caption(distribution_caption(forecast))
    st.info(decay_disclosure())


def _band_html(offset: float) -> str:
    """The 80% interval as one bar with the median marked, in plain HTML.

    No chart library: the whole figure is two divs, and adding a plotting dependency to the app
    container for a single horizontal bar would be a cold start paid on every page load.
    """
    return (
        "<div style='margin:0.25rem 0 0.75rem 0'>"
        "<div style='position:relative;height:10px;border-radius:5px;"
        "background:linear-gradient(90deg,#f3b0b0,#e6e6e6,#a7dca7)'>"
        f"<div style='position:absolute;left:{fixed(offset, 1)}%;top:-4px;width:2px;height:18px;"
        "background:#111827'></div></div></div>"
    )


def _render_trust(forecast: Mapping[str, Any] | None) -> None:
    """Open by default, because a number nobody has validated is the one a reader should doubt."""
    try:
        rows = model_evaluation.pooled_summary()
    except Exception as exc:  # noqa: BLE001 — a missing evaluation must not take the page with it
        with st.expander(TRUST_TITLE, expanded=False):
            st.caption(f"The evaluation table could not be read. {type(exc).__name__}: {exc}")
        return

    with st.expander(TRUST_TITLE, expanded=True):
        for line in model_evaluation.trust_lines(rows, forecast):
            st.write(line)
        st.caption(
            "Model names and the evaluation in full are in the "
            f"[project README]({model_evaluation.README_URL})."
        )


def _render_price_chart(ticker: str, prices: Sequence[Mapping[str, Any]]) -> None:
    st.subheader(f"{ticker} — {CHART_DAYS}-day price history")
    if not prices:
        st.warning(
            f"No rows in `{PRICES_TABLE}` for {ticker}. Run the ingestion and silver tasks first."
        )
        return
    st.line_chart(chart_data(prices), x="trade_date", y="close", height=280)
    last = prices[-1]
    st.caption(f"Last close {money(last.get('close'))} on {last.get('trade_date')}")


def _render_regime_detail(regime: Mapping[str, Any] | None) -> None:
    """The regime parameters, folded away. The headline badge is what most readers need.

    Collapsed rather than deleted: "0.31% mean daily return, 1.42% daily volatility" is the
    evidence behind the badge, and a page that shows a regime without ever showing its numbers is
    asking to be taken on faith.
    """
    with st.expander("Regime detail"):
        st.metric("Filtered probability", regime_headline(regime))
        if not regime:
            st.caption(
                "No `gold.regime_states` row. Either `fit_models` has not run for this ticker, or "
                "the ladder fell back to gbm, which has no regimes."
            )
            return
        st.caption(f"Model {regime.get('model_used', MISSING)} · as of {regime.get('as_of_date')}")
        st.write(regime_detail(regime))
        st.caption(
            f"News signal at the fit: {number(regime.get('current_news_signal'), 3)} "
            "(the decayed 3-day sentiment the model conditioned on)."
        )


def _render_news(news: Sequence[Mapping[str, Any]]) -> None:
    st.subheader("Recent news")
    st.caption(news_tone(news))
    for row in news:
        title = str(row.get("title") or "(untitled)")
        url = row.get("article_url")
        heading = f"[{title}]({url})" if url else title
        published = as_datetime(row.get("published_at"))
        when = age_phrase(published) if published else "at an unknown time"
        st.markdown(
            f"{sentiment_marker(row.get('sentiment_label'))} {heading}  \n"
            f"<span style='color:gray;font-size:0.85em'>"
            f"{row.get('publisher') or 'unknown publisher'} · {when} · "
            f"{row.get('sentiment_label') or 'unlabelled'}</span>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render()
