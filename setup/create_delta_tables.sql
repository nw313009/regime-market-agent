-- Delta table DDL (spec A-2, A-3, A-4, B-6). Keep in sync with sql/.
--
-- bronze.prices_raw       near-raw payload + source, ingested_at, request_id, ticker,
--                         source_timestamp.            MERGE (ticker, source_timestamp)
-- bronze.news_raw         near-raw payload + the same audit columns.
--                                                      MERGE (article_id, ticker)
-- bronze.ingestion_runs   run_id, task, started_at, finished_at, status, rows_written, error
--
-- silver.daily_prices     ticker, trade_date, open, high, low, close, volume, vwap.
--                                                      MERGE (ticker, trade_date)
-- silver.news_articles    article_id, ticker, published_at, title, description, publisher,
--                         sentiment_label, sentiment_score, embedding_text, article_url.
--                                                      MERGE (article_id, ticker)
--   MUST be created with Change Data Feed enabled, because the AI Search Delta Sync index
--   reads the CDF. Missing this at creation time is why an index comes back empty:
--     TBLPROPERTIES (delta.enableChangeDataFeed = true)
-- silver.daily_features   grain (ticker, trade_date); log_return, return_5d, momentum_5d,
--                         realized_vol_20d, volume_zscore_20d, news_sentiment_3d,
--                         news_count. Warm-up nulls stay in the table.
--
-- gold.regime_states      ticker, as_of_date, prob_low_vol, prob_high_vol, low/high mean and
--                         sigma, current_news_signal, model_used, model_version
-- gold.forecast_runs      forecast_id, ticker, generated_at, as_of_date, horizon_days,
--                         model_used, current_price, price_p10/50/90, return_p10/50/90,
--                         prob_positive, prob_loss_gt_5pct, prob_low_vol, prob_high_vol,
--                         n_paths, seed, model_version
-- gold.backtest_metrics   origin_date, ticker, model, brier, mae, covered_80 (bool),
--                         model_used, converged; plus a pooled_summary view/table carrying n
-- gold.model_calls        ts, task, model, latency_ms, ok, in_tokens, out_tokens
--
-- These tables are tiny (~2.5k rows per ticker). Do not partition them.
--
-- Run create_catalog.sql first. Every statement here is idempotent.

-- =====================================================================================
-- BRONZE (A-2) — implemented
-- =====================================================================================
-- Bronze is near-raw on purpose: it stores what the vendor returned plus the audit columns
-- needed to trace a row back to its request. No cleaning, no derived values, no session-date
-- mapping (that is silver's job), no sentiment_score (derived at silver build time, A-3).
--
-- Bronze does NOT enable Change Data Feed. Only silver.news_articles needs it, for the AI
-- Search Delta Sync index (C-1).

-- Massive daily aggregates. One row per (ticker, bar).
-- Payload mapping, VERIFIED live: o/h/l/c → open/high/low/close, v → volume, vw → vwap,
-- n → transactions, t → t_epoch_ms (and source_timestamp, the same instant as a TIMESTAMP).
CREATE TABLE IF NOT EXISTS market_intel.bronze.prices_raw (
  ticker            STRING    NOT NULL COMMENT 'Requested ticker. MERGE key 1.',
  source_timestamp  TIMESTAMP NOT NULL COMMENT 'Bar start from results[].t. 04:00Z == 00:00 America/New_York, i.e. the session open. MERGE key 2.',
  t_epoch_ms        BIGINT             COMMENT 'Raw results[].t, epoch-milliseconds, kept verbatim.',
  `open`            DOUBLE             COMMENT 'Raw o.',
  high              DOUBLE             COMMENT 'Raw h.',
  low               DOUBLE             COMMENT 'Raw l.',
  `close`           DOUBLE             COMMENT 'Raw c.',
  vwap              DOUBLE             COMMENT 'Raw vw (volume-weighted average price).',
  volume            DOUBLE             COMMENT 'Raw v. DOUBLE, not BIGINT: the vendor returns it in scientific notation (1.46147597081851e+08).',
  transactions      BIGINT             COMMENT 'Raw n, the trade count for the bar.',
  source            STRING    NOT NULL COMMENT 'Vendor identifier, always "massive".',
  ingested_at       TIMESTAMP NOT NULL COMMENT 'UTC wall-clock when this run wrote the row.',
  request_id        STRING             COMMENT 'request_id of the page envelope this row arrived in; ties the row to a vendor request.'
)
USING DELTA
COMMENT 'Near-raw Massive daily aggregates. MERGE on (ticker, source_timestamp) — never a blind INSERT.';

-- Massive news, exploded one row per (article, insight).
-- The explode is FROM insights, NOT from the tickers array (A-2/A-3): sentiment exists only
-- inside insights, so ticker and sentiment must be taken from the SAME insight. A ticker listed
-- in tickers with no insight therefore produces no row — a deliberate, accepted consequence.
-- The full raw tickers array is preserved in article_tickers so that decision stays auditable.
CREATE TABLE IF NOT EXISTS market_intel.bronze.news_raw (
  article_id             STRING    NOT NULL COMMENT 'Raw id, a stable 64-char hex digest. MERGE key 1.',
  ticker                 STRING    NOT NULL COMMENT 'insights[].ticker — the per-insight ticker, NOT a tickers[] element. MERGE key 2.',
  source_timestamp       TIMESTAMP NOT NULL COMMENT 'published_utc parsed to a UTC instant. Session mapping happens in silver.',
  published_utc          STRING             COMMENT 'Raw published_utc, ISO-8601 UTC string, kept verbatim.',
  title                  STRING             COMMENT 'Raw title.',
  description            STRING             COMMENT 'Raw description.',
  author                 STRING             COMMENT 'Raw author.',
  article_url            STRING             COMMENT 'Raw article_url.',
  image_url              STRING             COMMENT 'Raw image_url.',
  publisher_name         STRING             COMMENT 'publisher.name — publisher arrives as a nested dict.',
  publisher_homepage_url STRING             COMMENT 'publisher.homepage_url.',
  publisher_logo_url     STRING             COMMENT 'publisher.logo_url.',
  publisher_favicon_url  STRING             COMMENT 'publisher.favicon_url.',
  sentiment              STRING             COMMENT 'insights[].sentiment, RAW label (positive/neutral/negative). No numeric score exists in the payload; sentiment_score is derived in silver (A-3).',
  sentiment_reasoning    STRING             COMMENT 'insights[].sentiment_reasoning, raw text, kept for agent/UI display.',
  article_tickers        ARRAY<STRING>      COMMENT 'Raw tickers array. Insight tickers are normally a strict subset of this.',
  keywords               ARRAY<STRING>      COMMENT 'Raw keywords array.',
  source                 STRING    NOT NULL COMMENT 'Vendor identifier, always "massive".',
  ingested_at            TIMESTAMP NOT NULL COMMENT 'UTC wall-clock when this run wrote the row.',
  request_id             STRING             COMMENT 'request_id of the page envelope this row arrived in.'
)
USING DELTA
COMMENT 'Near-raw Massive news, one row per (article, insight). MERGE on (article_id, ticker) — never a blind INSERT.';

-- One row per ingestion task run, written even when the run fails.
-- error is redacted before it is stored: the API key travels as a query parameter, so raw
-- exception text can carry a credential into a queryable table (A-1 security rule).
CREATE TABLE IF NOT EXISTS market_intel.bronze.ingestion_runs (
  run_id       STRING    NOT NULL COMMENT 'UUID4 for this run. MERGE key.',
  task         STRING    NOT NULL COMMENT 'Workflow task name: ingest_prices | ingest_news.',
  started_at   TIMESTAMP          COMMENT 'UTC start.',
  finished_at  TIMESTAMP          COMMENT 'UTC end, set on success AND on failure.',
  status       STRING             COMMENT 'succeeded | failed.',
  rows_written BIGINT             COMMENT 'Rows merged into the bronze table by this run (post-dedupe staged count).',
  error        STRING             COMMENT 'Redacted exception summary when status = failed, else NULL. Never a response body or a full URL.'
)
USING DELTA
COMMENT 'Ingestion audit ledger (A-2). MERGE on (run_id).';

-- =====================================================================================
-- SILVER (A-3, A-4) — TODO: implement at checkpoint A-3.
-- Remember TBLPROPERTIES (delta.enableChangeDataFeed = true) on silver.news_articles at
-- CREATE time; adding it later does not backfill the feed the AI Search index reads.
-- =====================================================================================

-- =====================================================================================
-- GOLD (B-6, C-3) — TODO: implement at checkpoint B-6.
-- =====================================================================================
