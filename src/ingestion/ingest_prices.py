"""Price ingestion task: Massive aggregates -> ``bronze.prices_raw`` (spec A1.1, A-2).

For every ticker in the universe (the 5 seed tickers plus all watchlist tickers), fetch
daily aggregates since the last stored ``trade_date`` and MERGE into ``bronze.prices_raw``.

Bronze rows keep the near-raw payload plus ``source``, ``ingested_at``, ``request_id``,
``ticker`` and ``source_timestamp``.

MERGE keys: ``(ticker, source_timestamp)``. Never a blind INSERT — every write is
idempotent so a re-run produces identical row counts.

Also records the run in ``bronze.ingestion_runs``: ``run_id``, ``task``, ``started_at``,
``finished_at``, ``status``, ``rows_written``, ``error``.
"""
