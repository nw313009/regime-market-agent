"""Market Research page (spec A2, C-5).

Select a ticker, then show:

- Price history chart, from ``silver.daily_prices``.
- Current regime card, e.g. "High volatility - 73%", from the filtered probabilities in
  ``gold.regime_states``.
- Forecast distribution from ``gold.forecast_runs``: P10/P50/P90 of the 5-day return,
  P(positive), P(loss > 5%). Show the distribution, not a point prediction.
- Recent news from ``silver.news_articles`` with sentiment, plus ``news_count`` context so
  "no relevant news" reads differently from "neutral news".
- The news-decay assumption disclosure sentence, stated plainly rather than buried.

Read-only. No statistics are computed here; the page renders what Gold already contains.
"""
