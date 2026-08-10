"""Seed the minimum Lakebase state the demo needs (spec section 15).

Creates a demo user, a default watchlist, and the seed tickers from
``config.yaml -> tickers.seed`` (NVDA, MSFT, TSLA, AMZN, GOOGL) so the app has something to
show before anyone touches the watchlist.

Idempotent: running it twice must not duplicate rows. AMD is deliberately left out — adding
it live through the agent is the CDC demo moment.

TODO: implement.
"""
