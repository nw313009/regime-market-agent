# Databricks notebook source
"""A-0 smoke test — the FIRST implementation action, before any table is built.

Proves the Massive API key/plan works and that the caller has outbound connectivity to
Massive. Do not build around a source that has not been proven reachable.

Both routes below are VERIFIED live (200, same session): the aggregates route returned a
full 22-bar payload and the news route returned the documented envelope. Treat both
mappings as confirmed; do not "adjust" these paths from memory.

Dual auth, no secret ever written into this file (spec rule 5):

- Locally: MASSIVE_API_KEY from the environment, falling back to the repo-root .env.
- In a Databricks workspace cell: getpass prompt, so the key is typed rather than stored.
  For scheduled jobs the key comes from the secret scope instead::

      databricks secrets create-scope capstone
      databricks secrets put-secret capstone massive_api_key
      key = dbutils.secrets.get(scope="capstone", key="massive_api_key")

OUTPUT RULE (spec A-1): apiKey travels as a QUERY PARAMETER, so URLs and error bodies are
credential-bearing. This script prints the status code, the request_id, and structural facts
about the payload (key names and counts only) — never a response body, never a full URL.
A 401 body in particular can echo request params straight into a terminal log or an agent
transcript.

Reading the result:

- 200 -> proceed to A-1.
- 401 / 403 -> API key or plan problem. Stop and fix.
- Connection error -> egress problem. Stop and fix.

Anything other than 200 means stop and diagnose, not work around.
"""

import os
from getpass import getpass

import requests

BASE = "https://api.massive.com"
AGGS = f"{BASE}/v2/aggs/ticker/NVDA/range/1/day/2026-07-01/2026-08-01"
NEWS = f"{BASE}/v2/reference/news"


def get_api_key() -> str:
    """Environment, then repo-root .env, then an interactive prompt. Never a literal."""
    key = os.environ.get("MASSIVE_API_KEY", "")
    if key:
        return key

    try:
        from pathlib import Path

        from dotenv import load_dotenv

        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:  # notebook cell: no __file__
            root = Path.cwd()
        load_dotenv(dotenv_path=root / ".env")
        key = os.environ.get("MASSIVE_API_KEY", "")
    except ImportError:
        pass  # python-dotenv is local-only; absent on a cluster by design

    return key or getpass("MASSIVE_API_KEY: ")


def request_id_of(response) -> str:
    """Pull only request_id out of the payload. Never returns or logs the body itself."""
    try:
        return response.json().get("request_id", "n/a")
    except ValueError:
        return "unparseable"


def probe(label: str, url: str, key: str, **params):
    response = requests.get(url, params={"apiKey": key, **params}, timeout=30)
    print(f"[{label}] status={response.status_code} request_id={request_id_of(response)}")
    if response.status_code != 200:
        print(
            f"[{label}] non-200 — stop and diagnose. 401/403 = key or plan; connection error "
            "= egress. Body withheld on purpose: it can echo the apiKey query param."
        )
        return None
    return response.json()


def main() -> None:
    key = get_api_key()
    if not key:
        raise SystemExit("No MASSIVE_API_KEY available from env, .env, or prompt.")

    aggs = probe("aggregates", AGGS, key)
    if aggs is not None:
        print("  envelope keys :", sorted(aggs.keys()))
        print("  resultsCount  :", aggs.get("resultsCount"))
        bars = aggs.get("results") or []
        if bars:
            # Confirms the OHLCV field names the silver mapping depends on, incl. epoch-ms t.
            print("  bar fields    :", sorted(bars[0].keys()))

    news = probe("news", NEWS, key, ticker="NVDA", limit=3)
    if news is not None:
        print("  envelope keys :", sorted(news.keys()))
        articles = news.get("results") or []
        print("  articles      :", len(articles))
        if articles:
            print("  article fields:", sorted(articles[0].keys()))
            insights = articles[0].get("insights") or []
            if insights:
                # Must be exactly {ticker, sentiment, sentiment_reasoning} — no numeric score.
                print("  insight fields:", sorted(insights[0].keys()))


if __name__ == "__main__":
    main()
