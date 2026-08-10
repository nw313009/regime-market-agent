-- Unity Catalog bootstrap (spec section 15 — the environment must be recreatable from code).
--
-- Creates the market_intel catalog and its bronze / silver / gold schemas.
--
-- Layer responsibilities:
--   bronze  raw Massive payloads plus ingestion audit
--   silver  cleaned prices, normalized news, engineered daily features
--   gold    regime fits, forecasts, backtest metrics, model-call telemetry
--
-- The catalog name matches the `catalog` key in config/config.yaml (market_intel). SQL cannot
-- read the YAML, so these literals and that key must be changed together.
--
-- Run order: this file, then create_delta_tables.sql. Idempotent — safe to re-run.

CREATE CATALOG IF NOT EXISTS market_intel
  COMMENT 'Regime-aware market intelligence capstone (frozen architecture v2.1).';

CREATE SCHEMA IF NOT EXISTS market_intel.bronze
  COMMENT 'Near-raw Massive payloads plus ingestion audit. No cleaning, no derived columns.';

CREATE SCHEMA IF NOT EXISTS market_intel.silver
  COMMENT 'Cleaned prices, normalized news (one row per article-insight), daily features.';

CREATE SCHEMA IF NOT EXISTS market_intel.gold
  COMMENT 'Regime states, forecast runs, backtest metrics, model-call telemetry.';
