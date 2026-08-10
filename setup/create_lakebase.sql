-- Lakebase (Postgres) schema for application state (spec C-2). This file is the single source:
-- sql/ is intentionally empty (spec B0), and src/database/lakebase.py executes this file.
--
-- Instance: the capstone has its OWN Lakebase project, regime-market-database (Autoscaling,
-- Postgres branch production, endpoint primary). It does not share the instance hosting
-- ticket_system and weather_system.
--
-- Schema: tables still live in their own schema, market_system. Create the schema first, then
-- the four tables inside it, and fully qualify the schema in every query. A dedicated project
-- removes the collision risk but not the reason to qualify: search_path is not a contract,
-- qualification keeps grants and CDF targets unambiguous, and it matches the ticket_system /
-- weather_system convention.
--
--   CREATE SCHEMA IF NOT EXISTS market_system;
--
--   market_system.users(user_id, display_name, created_at)
--   market_system.watchlists(watchlist_id, user_id, name, created_at)
--   market_system.watchlist_tickers(watchlist_id, ticker, added_at, added_by,
--                                   PRIMARY KEY (watchlist_id, ticker))
--   market_system.research_reports(report_id, user_id, ticker, question, report_md,
--                                  forecast_id, created_at)
--
-- Lakebase is authoritative ONLY for application/transactional state. Analytical data stays
-- in Delta, and there is no Delta -> Lakebase forecast-serving sync in this capstone.
--
-- Enable Lakebase CDF on market_system.watchlist_tickers and market_system.research_reports so
-- their changes flow into Delta history tables — that is the CDC demo path. If the preview
-- toggle is unavailable in this workspace, fall back to the bootcamp's taught CDC method
-- (architecture doc section 20, condition 1).
--
-- SERVICE PRINCIPAL GRANTS — do this before the first app deploy, not while debugging it.
-- A new Databricks App gets a new service principal, and that identity has no Postgres role
-- yet. Create its role through the regime-market-database project's OAuth tab, then grant it USAGE on
-- schema market_system and the needed privileges on these four tables. Missing grants surface
-- at first deploy as an authentication or permission failure that looks like an application
-- bug and is not.
--
-- Connections are made with psycopg v3 through the pooled, per-connection-OAuth pattern in
-- db.py (see src/database/lakebase.py). No static password exists anywhere.
--
-- POSTGRES DIALECT, not Delta: TEXT and TIMESTAMPTZ, real PRIMARY KEY and FOREIGN KEY
-- constraints, ON CONFLICT upserts in the application code. Every statement is idempotent, and
-- src.database.lakebase.ensure_tables() executes THIS FILE rather than a second copy of these
-- statements — the same single-source rule that emptied sql/ (spec B0).
--
-- IDs ARE APPLICATION-GENERATED TEXT, not serials or gen_random_uuid() defaults. save_report has
-- to return the id it just wrote (spec C-4), and generating it in Python keeps that a plain
-- INSERT instead of an INSERT ... RETURNING whose value the tests could only observe through a
-- live database. It also lets the demo user and watchlist have readable, stable ids.

CREATE SCHEMA IF NOT EXISTS market_system;

-- One row per person using the app. The demo seeds exactly one (src.database.lakebase.seed_demo).
CREATE TABLE IF NOT EXISTS market_system.users (
  user_id      TEXT        PRIMARY KEY,
  display_name TEXT        NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A named list of tickers belonging to a user. The demo has one; the schema does not assume it.
CREATE TABLE IF NOT EXISTS market_system.watchlists (
  watchlist_id TEXT        PRIMARY KEY,
  user_id      TEXT        NOT NULL REFERENCES market_system.users (user_id) ON DELETE CASCADE,
  name         TEXT        NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Membership. The composite PRIMARY KEY is what makes add_ticker idempotent: the INSERT carries
-- ON CONFLICT (watchlist_id, ticker) DO NOTHING, so adding AMD twice is a no-op, not an error.
-- This is the table the CDC demo watches (the agent adds AMD here, and the change arrives in
-- Delta through Lakebase CDF).
CREATE TABLE IF NOT EXISTS market_system.watchlist_tickers (
  watchlist_id TEXT        NOT NULL REFERENCES market_system.watchlists (watchlist_id) ON DELETE CASCADE,
  ticker       TEXT        NOT NULL,
  added_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  added_by     TEXT        REFERENCES market_system.users (user_id),
  PRIMARY KEY (watchlist_id, ticker)
);

-- Saved agent answers. forecast_id points at market_intel.gold.forecast_runs.forecast_id, which
-- is a uuid5 of (ticker, as_of_date, model_used) and therefore survives a re-run of the daily job
-- (spec B-6). It is deliberately NOT a foreign key: that row lives in Delta, in the other store,
-- and the two never join in the database.
CREATE TABLE IF NOT EXISTS market_system.research_reports (
  report_id   TEXT        PRIMARY KEY,
  user_id     TEXT        NOT NULL REFERENCES market_system.users (user_id) ON DELETE CASCADE,
  ticker      TEXT        NOT NULL,
  question    TEXT,
  report_md   TEXT        NOT NULL,
  forecast_id TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS research_reports_user_ticker_idx
  ON market_system.research_reports (user_id, ticker, created_at DESC);

-- REPLICA IDENTITY FULL ON THE TWO CDC TABLES — this is the Lakebase CDF requirement, not a
-- preference. Postgres logical decoding emits only the primary key for an UPDATE or DELETE under
-- the default replica identity, so a delete arriving in the Delta history table would carry no
-- ticker and no added_by, and the "show the change arriving in Delta" step of the demo would show
-- a row of nulls. FULL puts the whole old tuple in the WAL.
--
-- Requires table ownership. If this errors, the role running the file is not the table owner —
-- fix the grant, do not drop the statement.
ALTER TABLE market_system.watchlist_tickers  REPLICA IDENTITY FULL;
ALTER TABLE market_system.research_reports   REPLICA IDENTITY FULL;
