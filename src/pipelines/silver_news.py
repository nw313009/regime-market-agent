"""``bronze.news_raw`` -> ``silver.news_articles`` (spec A-3).

Schema: ``article_id``, ``ticker``, ``published_at``, ``title``, ``description``,
``publisher``, ``sentiment_label``, ``sentiment_score``, ``embedding_text``,
``article_url``.

- ``embedding_text`` = ``title + "\\n" + description``. It is the embedding source column
  for the AI Search Delta Sync index.
- Composite key ``(article_id, ticker)``: explode the article's tickers array so there is
  one row per (article, ticker). MERGE on that composite key.
- Change Data Feed must be enabled at table creation, because the AI Search index reads
  the Delta CDF::

      TBLPROPERTIES (delta.enableChangeDataFeed = true)

  An index that is empty after a sync usually means this property was missing when the
  table was created.
- Normalize the vendor sentiment label to the {positive, neutral, negative} vocabulary the
  feature pipeline maps to {+1, 0, -1}.
"""
