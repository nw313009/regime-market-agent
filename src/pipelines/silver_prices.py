"""``bronze.prices_raw`` -> ``silver.daily_prices`` (spec A-3).

Schema: ``ticker``, ``trade_date``, ``open``, ``high``, ``low``, ``close``, ``volume``,
``vwap``.

MERGE on ``(ticker, trade_date)``.

Timestamp rule: Massive returns epoch-milliseconds. Convert to the trading date in the
exchange timezone (``America/New_York``), NOT a UTC-naive date. A UTC-naive conversion
silently shifts late-session bars onto the wrong trading day.
"""
