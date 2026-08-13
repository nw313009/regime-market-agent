"""Streamlit Databricks App entry point (spec C-5).

Three pages live in ``app/pages/``: Market Research, Research Agent, Model Evaluation. Streamlit
discovers them from the directory; this file is the landing page and the one place that configures
the app shell.

Data access:

- Delta through ``databricks-sql-connector`` against a serverless SQL warehouse
  (``src/database/delta.py``).
- Lakebase through the psycopg v3 pool in ``src/database/lakebase.py``, which authenticates each
  connection with a short-lived OAuth credential rather than a stored password. THE APP CONTAINER
  IS THE ONLY PLACE THAT MAY IMPORT THAT MODULE — psycopg aborts a serverless kernel at import
  (see ``requirements-databricks.txt``), which is why the workflow reaches Postgres over JDBC.

There is no SparkSession in a Databricks App, which is why the app reads the small Gold result set
over SQL rather than through Spark, and why ``app.yaml`` sets ``TELEMETRY_MODE=log``: the Delta
telemetry writer needs a session the app does not have.

Reads are cached with ``st.cache_data(ttl=600)`` — see ``app/common.py``.

"App can't read Delta" is a warehouse id or permissions problem in ``app.yaml``, not a code
problem — check there first (spec C-e).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402 — the path has to be set before the app imports

from app.common import config, decay_disclosure, seed_tickers  # noqa: E402

TITLE = "Regime-Aware Market Intelligence"


def render() -> None:
    """The landing page: what this system is, and what each page will and will not tell you."""
    st.set_page_config(page_title=TITLE, page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title(TITLE)

    cfg = config()
    st.caption(
        f"Catalog `{cfg['catalog']}` · seed universe {', '.join(seed_tickers(cfg))} · "
        f"{cfg['forecast']['horizon_days']}-day horizon, "
        f"{cfg['forecast']['n_paths']:,} simulated paths"
    )

    st.markdown(
        """
This system fits a two-regime Markov switching model to each ticker's daily returns, lets news
sentiment move the probability of switching between the calm and turbulent regimes, and simulates
the next five sessions from the fitted parameters. Everything the pages show is read from the Gold
tables that pipeline produces.

**Market Research** — the current regime, the forecast distribution, and the recent news behind it.

**Research Agent** — ask why, in words. The agent reads the same Gold numbers and the news index;
it never computes a statistic of its own and it never gives advice.

**Model Evaluation** — whether the news-aware model actually beats the simpler ones, with the
sample size and the fallback rate shown next to every score.
"""
    )

    st.info(decay_disclosure())

    st.warning(
        "Research and demonstration only. Nothing here is investment advice, and a five-day "
        "forecast from a two-regime model is a description of recent volatility rather than a "
        "prediction anyone should trade on."
    )


if __name__ == "__main__":
    render()
