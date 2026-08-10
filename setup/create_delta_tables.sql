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
--                         sentiment_label, sentiment_score, sentiment_reasoning,
--                         embedding_text, article_url.
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

-- One row per task run, written even when the run fails.
-- Introduced at A-2 for the ingestion tasks and extended at A-4 to the pipeline tasks, so a
-- workflow run leaves one audit row per task whatever that task does.
-- error is redacted before it is stored: the API key travels as a query parameter, so raw
-- exception text can carry a credential into a queryable table (A-1 security rule).
CREATE TABLE IF NOT EXISTS market_intel.bronze.ingestion_runs (
  run_id       STRING    NOT NULL COMMENT 'UUID4 for this run. MERGE key.',
  task         STRING    NOT NULL COMMENT 'Task name: ingest_prices | ingest_news | build_silver_prices | build_silver_news | build_features.',
  started_at   TIMESTAMP          COMMENT 'UTC start.',
  finished_at  TIMESTAMP          COMMENT 'UTC end, set on success AND on failure.',
  status       STRING             COMMENT 'succeeded | failed.',
  rows_written BIGINT             COMMENT 'Rows written by this run: rows merged into the target table (post-dedupe staged count).',
  error        STRING             COMMENT 'Redacted exception summary when status = failed, else NULL. Never a response body or a full URL.'
)
USING DELTA
COMMENT 'Run audit ledger for ingestion (A-2) and pipeline (A-4) tasks. MERGE on (run_id).';

-- =====================================================================================
-- SILVER (A-3) — implemented
-- =====================================================================================
-- Silver is where cleaning and derivation happen: the session-date mapping, the volume cast,
-- the sentiment_score mapping and embedding_text. Both tables are rebuilt from bronze by a
-- MERGE on their declared keys, so a re-run is idempotent.

-- One row per (ticker, trading session). trade_date is the bronze bar instant resolved in
-- America/New_York, never a UTC-naive date: a UTC truncation moves a winter bar stamped at
-- 04:00Z onto the previous session.
CREATE TABLE IF NOT EXISTS market_intel.silver.daily_prices (
  ticker     STRING NOT NULL COMMENT 'MERGE key 1.',
  trade_date DATE   NOT NULL COMMENT 'Exchange session date (America/New_York) of the bar. MERGE key 2.',
  `open`     DOUBLE          COMMENT 'Session open.',
  high       DOUBLE          COMMENT 'Session high.',
  low        DOUBLE          COMMENT 'Session low.',
  `close`    DOUBLE          COMMENT 'Session close. The modeling layer takes log returns of this column.',
  volume     BIGINT          COMMENT 'Shares traded, rounded from the bronze DOUBLE: the vendor sends a fractional value in scientific notation, and share counts are whole.',
  vwap       DOUBLE          COMMENT 'Volume-weighted average price.'
)
USING DELTA
COMMENT 'Cleaned daily bars, grain (ticker, trade_date). MERGE on (ticker, trade_date) — never a blind INSERT.';

-- One row per (article, insight), carried through from bronze. sentiment_score is DERIVED here.
-- CHANGE DATA FEED IS MANDATORY AT CREATION TIME. The AI Search Delta Sync index (C-1) reads
-- this table's CDF; enabling the property later does not backfill the feed, and the symptom is
-- an index that syncs to zero rows.
CREATE TABLE IF NOT EXISTS market_intel.silver.news_articles (
  article_id          STRING NOT NULL COMMENT 'Bronze article_id, a stable 64-char hex digest. MERGE key 1.',
  ticker              STRING NOT NULL COMMENT 'Per-insight ticker from bronze. MERGE key 2.',
  published_at        TIMESTAMP       COMMENT 'published_utc parsed from its ISO-8601 UTC string. Mapping to a trading session happens in A-4, not here.',
  title               STRING          COMMENT 'Passed through from bronze.',
  description         STRING          COMMENT 'Passed through from bronze. May be NULL.',
  publisher           STRING          COMMENT 'publisher.name, flattened in bronze as publisher_name.',
  sentiment_label     STRING          COMMENT 'RAW vendor label, kept verbatim even when unrecognized.',
  sentiment_score     INT             COMMENT 'DERIVED here: positive=+1, neutral=0, negative=-1, unrecognized=0 with a logged warning. daily_features.s_t consumes this unchanged.',
  sentiment_reasoning STRING          COMMENT 'Raw vendor reasoning text, kept for agent and UI display.',
  embedding_text      STRING          COMMENT 'title + newline + description, description omitted when absent. Embedding source column for the AI Search index.',
  article_url         STRING          COMMENT 'Passed through from bronze.'
)
USING DELTA
COMMENT 'Normalized news, one row per (article, insight). MERGE on (article_id, ticker) — never a blind INSERT.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- =====================================================================================
-- SILVER FEATURES (A-4) — implemented
-- =====================================================================================
-- The contract between Spark and the modeling layer: models read this table for one ticker with a
-- single .toPandas() and never touch bronze.
--
-- Grain is (ticker, trade_date) on XNYS trading days only. Rows whose rolling features are still
-- warming up KEEP THEIR NULLS and stay in the table — dropping them here would hide how much
-- history a feature needs, and the modeling layer drops them at read time instead.
--
-- Every rolling column is NULL until its window is genuinely full, because Spark aggregates skip
-- NULLs and would otherwise report a 19-observation standard deviation as a 20-day one. That puts
-- the first non-null at a known row: momentum_5d at row 5, realized_vol_20d at row 20, and
-- volume_zscore_20d at row 19, one row earlier because volume has no undefined first row.
CREATE TABLE IF NOT EXISTS market_intel.silver.daily_features (
  ticker            STRING NOT NULL COMMENT 'MERGE key 1.',
  trade_date        DATE   NOT NULL COMMENT 'XNYS session date. MERGE key 2.',
  `close`           DOUBLE          COMMENT 'Session close, carried from daily_prices. The forecast layer reads the last value as current_price (B-4), so it must not require a second table.',
  volume            BIGINT          COMMENT 'Shares traded, carried from daily_prices as the volume_zscore_20d input.',
  log_return        DOUBLE          COMMENT 'ln(close / previous close). NULL on a ticker first row. The regime model estimates on this column, rescaled to percent (x100).',
  return_5d         DOUBLE          COMMENT 'close / close 5 sessions ago - 1. NULL for the first 5 rows.',
  momentum_5d       DOUBLE          COMMENT 'Sum of the last 5 log_returns. NULL until 5 non-null returns exist (row 5).',
  realized_vol_20d  DOUBLE          COMMENT 'stddev_samp(log_return) over the trailing 20 rows. NULL until 20 non-null returns exist (row 20).',
  volume_zscore_20d DOUBLE          COMMENT '(volume - mean_20) / stddev_20 over the trailing 20 rows including this one. NULL until row 19, and NULL when the window is constant (zero deviation).',
  s_t               DOUBLE          COMMENT 'Mean sentiment_score of the articles assigned to this session, 0.0 when the session had none. Assignment rolls a closed-market timestamp FORWARD to the next XNYS session via the calendar, never by weekday arithmetic: an article published on Good Friday belongs to the following Monday.',
  news_sentiment_3d DOUBLE          COMMENT 'Decayed sentiment (1.0*s_t + 0.5*s_t-1 + 0.25*s_t-2) / 1.75. Missing lags count as 0.0 because no news is a real zero. This is the exog_tvtp input, and the model shifts it one trading day before use (architecture 5).',
  news_count        BIGINT          COMMENT 'Articles assigned to this session, 0 when none. Preserved so a sentiment value can be weighed against how much news produced it.'
)
USING DELTA
COMMENT 'Model-ready features, grain (ticker, trade_date) on XNYS sessions. MERGE on (ticker, trade_date) — never a blind INSERT.';

-- =====================================================================================
-- GOLD (B-6, C-3) — TODO: implement at checkpoint B-6.
-- =====================================================================================
