"""AI Search setup (spec C-1).

Creates:

- One vector search endpoint, STANDARD tier.
- A Delta Sync index ``market_intel.silver.news_index`` over ``silver.news_articles``, with
  ``embedding_text`` as the embedding source column (managed embeddings), sync mode
  TRIGGERED.

The ``sync_news_index`` workflow task triggers the sync; it runs last in the daily job.

Query path used by ``search_market_news``: hybrid search, filtered by ticker, top_k around 5.

``silver.news_articles`` must have had ``delta.enableChangeDataFeed = true`` set at creation
time, since the Delta Sync index reads the CDF. An index that is empty after a sync is this
property missing, not a sync bug.

Financial news text is the project's required unstructured-data path. Retrieval supplies
evidence; it never generates the numerical forecast.
"""
