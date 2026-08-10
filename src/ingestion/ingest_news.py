"""News ingestion task: Massive news -> ``bronze.news_raw`` (spec A1.2, A-2).

For every ticker in the universe, fetch news since the last stored ``published_at`` and
MERGE into ``bronze.news_raw``.

Bronze rows keep the near-raw payload plus ``source``, ``ingested_at``, ``request_id``,
``ticker`` and ``source_timestamp``.

MERGE keys: ``(article_id, ticker)`` after a light explode of the article's tickers array.
One article associated with several companies therefore yields one bronze row per ticker,
which preserves ticker-specific sentiment downstream.

Also records the run in ``bronze.ingestion_runs``.
"""
