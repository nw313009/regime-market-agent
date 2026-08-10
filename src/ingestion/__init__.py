"""Massive API ingestion into the bronze layer.

Owns the only outbound network calls to the market-data vendor and the only writes to
``bronze.prices_raw``, ``bronze.news_raw`` and ``bronze.ingestion_runs``.
"""
