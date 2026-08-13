"""Market Research page (spec A2, C-5).

Select a ticker; see its price history, the regime the model currently believes it is in, the
five-day forecast distribution, and the news that fed it. Every number is READ from Gold. This
page computes no statistics — if a number is not in ``gold.regime_states`` or
``gold.forecast_runs``, it is not shown.

TWO THINGS ARE STATED EVEN WHEN THEY ARE INCONVENIENT:

- The news-decay assumption, next to the forecast rather than in a footer (spec A2). The model
  conditions on today's news and decays it; it does not know tomorrow's.
- "No relevant news" as distinct from "neutral news". A ticker with an empty window and a ticker
  the press is indifferent about produce the same sentiment number and mean completely different
  things, and the tone line says which one this is.

A GBM DAY HAS NO REGIME CARD. When the ladder falls back to Model A there are no regimes, the
daily task writes no ``regime_states`` row, and the page says so instead of rendering zeros.

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
    age_phrase,
    catalog,
    decay_disclosure,
    money,
    number,
    pct,
    seed_tickers,
)
from src.database import delta  # noqa: E402

TITLE = "Market Research"

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

    _render_price_chart(ticker, prices)
    left, right = st.columns(2)
    with left:
        _render_regime(regime)
    with right:
        _render_forecast(forecast)
    _render_news(news)


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


def _render_regime(regime: Mapping[str, Any] | None) -> None:
    st.subheader("Current regime")
    st.metric("Filtered probability", regime_headline(regime))
    if not regime:
        st.caption(
            "No `gold.regime_states` row. Either `fit_models` has not run for this ticker, or the "
            "ladder fell back to gbm, which has no regimes."
        )
        return
    st.caption(f"Model {regime.get('model_used', MISSING)} · as of {regime.get('as_of_date')}")
    st.write(regime_detail(regime))
    st.caption(
        f"News signal at the fit: {number(regime.get('current_news_signal'), 3)} "
        "(the decayed 3-day sentiment the model conditioned on)."
    )


def _render_forecast(forecast: Mapping[str, Any] | None) -> None:
    st.subheader("Five-day forecast")
    if not forecast:
        st.caption(forecast_caption(None))
        return

    low, mid, high = st.columns(3)
    low.metric("P10 return", pct(forecast.get("return_p10"), 1, signed=True))
    mid.metric("P50 return", pct(forecast.get("return_p50"), 1, signed=True))
    high.metric("P90 return", pct(forecast.get("return_p90"), 1, signed=True))

    up, down = st.columns(2)
    up.metric("P(return > 0)", pct(forecast.get("prob_positive"), 0))
    down.metric("P(loss > 5%)", pct(forecast.get("prob_loss_gt_5pct"), 0))

    st.caption(forecast_caption(forecast))
    st.caption(
        f"Price range {money(forecast.get('price_p10'))} – {money(forecast.get('price_p90'))} "
        f"from {money(forecast.get('current_price'))}."
    )
    st.info(decay_disclosure())


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
