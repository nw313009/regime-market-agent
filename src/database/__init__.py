"""Data access. Two stores, two responsibilities, no loop between them.

- Delta / Unity Catalog is authoritative for all analytical data: raw and cleaned prices,
  normalized news, engineered features, regime fits, forecasts, backtest results.
- Lakebase is authoritative only for application state: users, watchlists, watchlist
  membership, saved research reports.

Operational writes flow Lakebase -> Lakebase CDF -> Delta history. Analytical data flows the
other way. There is no Delta -> Lakebase forecast-serving synchronization in this capstone;
the app reads the small Gold result set directly.
"""
