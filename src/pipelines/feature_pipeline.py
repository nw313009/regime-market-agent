"""silver -> ``silver.daily_features`` (spec A-4).

Grain: ``(ticker, trade_date)``, trading days only, from ``exchange_calendars`` calendar
``XNYS``.

Columns::

    log_return        = ln(close / lag(close))
    return_5d         = close / lag(close, 5) - 1
    momentum_5d       = sum of the last 5 log_returns
    realized_vol_20d  = stddev_samp(log_return) over the trailing 20 rows
    volume_zscore_20d = (volume - mean_20) / stddev_20
    s_t               = mean of per-article normalized sentiment for that session
                        (positive -> +1, neutral -> 0, negative -> -1); 0 if no articles
    news_sentiment_3d = (1.0*s_t + 0.5*s_{t-1} + 0.25*s_{t-2}) / 1.75
    news_count        = number of articles mapped to that session

News-to-session assignment: map ``published_at`` to its trading session; if the market was
closed, assign the NEXT session taken from the market calendar. Never weekday arithmetic —
that gets holidays wrong. Saturday and Sunday articles land in Monday's ``s_t`` and
``news_count``.

``news_count`` exists so the UI can distinguish "no relevant news" from "neutral news":
``s_t = 0`` is ambiguous on its own.

Implementation: Spark window functions partitioned by ``ticker`` ordered by ``trade_date``.
Rows whose rolling features are still null (the warm-up period) STAY in the table; dropping
warm-up NaNs is the modeling layer's job, not this pipeline's.
"""
