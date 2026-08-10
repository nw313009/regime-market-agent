"""``bronze.news_raw`` -> ``silver.news_articles`` (spec A-3).

Payload shapes here are VERIFIED against a live Massive response, not inferred.

Schema: ``article_id``, ``ticker``, ``published_at``, ``title``, ``description``,
``publisher``, ``sentiment_label``, ``sentiment_score``, ``sentiment_reasoning``,
``embedding_text``, ``article_url``.

EXPLODE FROM ``insights``, NOT ``tickers``. Each article carries both a ``tickers`` array and
an ``insights`` array of ``{ticker, sentiment, sentiment_reasoning}``. Sentiment exists only
inside ``insights``, so the explode produces one row per (article, insight), taking BOTH the
ticker and its sentiment from the insight.

Deliberate, accepted consequence: a ticker listed in ``tickers`` with no corresponding insight
produces NO row. Exploding ``tickers`` instead would manufacture rows with null sentiment.
Observed live, ``insights`` tickers were a subset of ``tickers`` in 10/10 articles with zero
mismatches, so a strict subset is normal input, not an anomaly to repair.

Field mapping::

    article_id          <- id                  (stable 64-char hex digest)
    published_at        <- published_utc       (ISO-8601 UTC, e.g. "2026-08-10T02:15:00Z")
    publisher           <- publisher.name      (publisher is a nested dict:
                                                name/homepage_url/logo_url/favicon_url)
    ticker              <- insights[].ticker
    sentiment_label     <- insights[].sentiment              (raw: positive/neutral/negative)
    sentiment_reasoning <- insights[].sentiment_reasoning    (raw text, for agent/UI display)
    title, description, article_url pass through unchanged.

``sentiment_score`` is DERIVED here, not ingested. Massive returns no numeric score — insight
keys are exactly ``{ticker, sentiment, sentiment_reasoning}``. Map positive to +1, neutral to
0, negative to -1; any unrecognized label maps to 0 AND logs a warning, so a new vendor label
degrades to neutral rather than failing the build or vanishing silently. ``daily_features.s_t``
consumes ``sentiment_score`` unchanged (A-4).

``embedding_text`` = ``title + "\\n" + description``. It is the embedding source column for the
AI Search Delta Sync index.

MERGE on the composite key ``(article_id, ticker)``.

Change Data Feed must be enabled at table creation, because the AI Search index reads the Delta
CDF::

    TBLPROPERTIES (delta.enableChangeDataFeed = true)

An index that is empty after a sync usually means this property was missing when the table was
created.

Timestamps: ``published_utc`` is an ISO-8601 string, NOT epoch-milliseconds — that applies to
the aggregates ``t`` field instead. Parse accordingly, then map to the trading session in
America/New_York.
"""
