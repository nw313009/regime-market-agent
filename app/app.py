"""Streamlit Databricks App entry point (spec C-5).

Three pages live in ``app/pages/``: Market Research, Research Agent, Model Evaluation.

Data access:

- Delta through ``databricks-sql-connector`` against a serverless SQL warehouse.
- Lakebase through the psycopg v3 pool in ``src/database/lakebase.py``, which authenticates
  each connection with a short-lived OAuth credential rather than a stored password.

There is no SparkSession in a Databricks App, which is why the app reads the small Gold
result set over SQL rather than through Spark.

Cache reads with ``st.cache_data(ttl=600)``.

"App can't read Delta" is a warehouse id or permissions problem in ``app.yaml``, not a code
problem — check there first.
"""
