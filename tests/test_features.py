"""Feature-pipeline tests (spec A-5).

- ``test_features``: feed a known synthetic price series through the pipeline and assert the
  EXACT expected ``log_return``, ``realized_vol_20d`` and ``volume_zscore_20d``. Exact values
  on a hand-computable series, not approximate smoke checks.
- ``test_weekend_news``: Saturday and Sunday articles must land in Monday's ``s_t`` and
  ``news_count``. Extend this to a holiday to prove the market calendar is being used rather
  than weekday arithmetic.

Also covers the B-7 ``test_log_returns`` requirement: exact log-return values on synthetic
prices.

FIXTURES must mirror the REAL Massive payload shapes for both endpoints — no invented schemas.
Fixtures are the only place the vendor contract is pinned in code, so a fixture that disagrees
with production is worse than none: it makes wrong code pass. Required for news:

- ``publisher`` as a nested dict; assert the mapping takes ``publisher.name``.
- ``id`` as a 64-char hex digest, mapping to ``article_id``.
- ``published_utc`` as an ISO-8601 UTC string, not epoch-ms.
- ``insights`` entries carrying exactly ``{ticker, sentiment, sentiment_reasoning}`` with NO
  numeric score.
- At least one article where ``insights`` is a STRICT subset of ``tickers``, asserting the extra
  ticker yields NO row. That is the A-3 explode rule and the one that silently regresses if
  someone "simplifies" the explode back to the ``tickers`` array.
- One article with an unrecognized sentiment label, asserting ``sentiment_score`` -> 0 plus a
  logged warning.

TODO: implement.
"""
