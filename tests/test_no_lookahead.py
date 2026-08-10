"""Look-ahead / leakage tests (spec B-7). The tests that decide whether the backtest means
anything.

- ``test_tvtp_no_lookahead``: fit and forecast at origin T. Then corrupt every
  ``news_sentiment_3d`` value AFTER T with random values, refit and forecast again, and assert
  the ``ForecastSummary`` is bit-identical. If it changed, future news reached the fit — the
  ``shift(1)`` alignment in ``news_markov.py`` is wrong.
- Grep-level assertion: no reference to ``smoothed_marginal_probabilities`` anywhere under
  ``src/models/``. Smoothed probabilities use the full sample; reading them makes a backtest
  look excellent and mean nothing. A static check is used here because the numerical
  difference is easy to miss by eye.
- Extend the same corrupt-the-future idea to prices: corrupting closes after T must not change
  the forecast at T.

If the backtest ever comes back wildly good, these two tests are the first thing to check.

TODO: implement.
"""
