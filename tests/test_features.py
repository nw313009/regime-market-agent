"""Feature-pipeline tests (spec A-5).

- ``test_features``: feed a known synthetic price series through the pipeline and assert the
  EXACT expected ``log_return``, ``realized_vol_20d`` and ``volume_zscore_20d``. Exact values
  on a hand-computable series, not approximate smoke checks.
- ``test_weekend_news``: Saturday and Sunday articles must land in Monday's ``s_t`` and
  ``news_count``. Extend this to a holiday to prove the market calendar is being used rather
  than weekday arithmetic.

Also covers the B-7 ``test_log_returns`` requirement: exact log-return values on synthetic
prices.

TODO: implement.
"""
