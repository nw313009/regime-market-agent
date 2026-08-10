# Databricks notebook source
"""A-0 smoke test — the FIRST implementation action, before any table is built.

Proves two things at once: the Massive API key/plan works, and the workspace has outbound
connectivity to Massive. Do not build around a source that has not been proven reachable.

Prerequisite (run once from the CLI)::

    databricks secrets create-scope capstone
    databricks secrets put-secret capstone massive_api_key

The request, near-verbatim from the spec::

    import requests
    key = dbutils.secrets.get(scope="capstone", key="massive_api_key")
    r = requests.get(f"{BASE_URL}/v2/aggs/ticker/NVDA/range/1/day/2026-07-01/2026-08-01",
                     params={"apiKey": key}, timeout=30)
    print(r.status_code); print(r.json() if r.ok else r.text[:500])

Adjust the path to Massive's CURRENT aggregates route — check their docs, do not trust
memory.

Reading the result:

- 200 plus JSON -> proceed to A-1.
- 401 / 403     -> API key or plan problem. Stop and fix.
- Connection error -> workspace egress problem. Stop and fix.

Anything other than 200 means stop and diagnose, not work around.

This notebook is a thin wrapper, like every notebook here: %pip install -r requirements.txt,
append the repo root to sys.path, import from src/, call functions. All logic lives in
src/*.py so it is testable locally with pytest without a cluster.
"""
