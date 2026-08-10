"""Regime-Aware Market Intelligence Agent — source package (architecture v2.1, frozen).

Layer boundaries are architectural, not stylistic:

- ``ingestion`` and ``pipelines`` run on Spark (Massive API -> bronze -> silver -> features).
- ``models`` runs in pandas/statsmodels only and MUST NOT import pyspark. The single
  crossing point is one ``.toPandas()`` call at the ``silver.daily_features`` boundary.
- ``agent``, ``llm`` and ``database`` serve the application; they read Gold and write
  Lakebase, and never compute statistics.
"""
