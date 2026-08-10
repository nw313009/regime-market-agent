"""Idempotency tests (spec A-5, and rule 6: every pipeline write is idempotent).

- ``test_idempotency``: run the silver build twice over the same bronze data and assert
  identical row counts AND identical checksums. Row counts alone would miss a MERGE that
  updates rows it should have left untouched.
- Extend the same double-run assertion to the bronze ingestion tasks and to the feature
  pipeline.
- Assert every write path is a MERGE on the declared keys, never a blind INSERT:

    bronze.prices_raw       (ticker, source_timestamp)
    bronze.news_raw         (article_id, ticker)
    silver.daily_prices     (ticker, trade_date)
    silver.news_articles    (article_id, ticker)
    silver.daily_features   (ticker, trade_date)

This matters operationally: the daily job retries, and a retry must not duplicate data.

TODO: implement.
"""
