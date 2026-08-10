"""Delta / Unity Catalog access helpers.

Two distinct access paths, deliberately:

- Pipeline code (ingestion, silver, features) reads and writes through Spark.
- The Streamlit app reads through ``databricks-sql-connector`` against a serverless SQL
  warehouse, because a Databricks App has no SparkSession.

Write rule for every pipeline table: MERGE on the declared keys, never a blind INSERT, so
re-running a task is idempotent (spec rule 6).

Declared MERGE keys:

- ``bronze.prices_raw``     -> ``(ticker, source_timestamp)``
- ``bronze.news_raw``       -> ``(article_id, ticker)``
- ``silver.daily_prices``   -> ``(ticker, trade_date)``
- ``silver.news_articles``  -> ``(article_id, ticker)``
- ``silver.daily_features`` -> ``(ticker, trade_date)``

The catalog name comes from config (``catalog: market_intel``), never hard-coded at call
sites.

These tables are tiny (roughly 2.5k rows per ticker). Do not partition them; the defaults
are correct.
"""
