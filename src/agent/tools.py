"""The four agent tools, each with a JSON-schema declaration (spec C-4).

Read tools (numerical and retrieval context)::

    get_market_forecast(ticker)
        -> the latest gold.forecast_runs row joined with the gold.regime_states row

    search_market_news(ticker, query, k=5)
        -> AI Search hybrid results, filtered by ticker

Write tools (Lakebase state)::

    update_watchlist(action: "add" | "remove", ticker)
        -> performs the Lakebase write, returns the new watchlist

    save_research_report(ticker, question, report_md)
        -> performs the Lakebase write, returns the new report id

The read tools return the numbers; they do not compute them. Retrieval does not generate the
forecast.

Write tools are the CDC demo path: the row lands in Lakebase and then arrives in the Delta
history table through Lakebase CDF.
"""
