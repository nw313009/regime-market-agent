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
--                         embedding_text, article_url, doc_id.
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
--                                                      MERGE (ticker, as_of_date)
-- gold.forecast_runs      forecast_id, ticker, generated_at, as_of_date, horizon_days,
--                         model_used, current_price, price_p10/50/90, return_p10/50/90,
--                         prob_positive, prob_loss_gt_5pct, prob_low_vol, prob_high_vol,
--                         n_paths, seed, model_version
--                                                      MERGE (ticker, as_of_date, model_used)
-- gold.backtest_metrics   origin_date, ticker, model, brier, mae, covered_80 (bool),
--                         model_used, converged, failure_reason + the inputs behind the scores
--                                                      MERGE (origin_date, ticker, model)
-- gold.backtest_summary   the pooled summary carrying n: model, n, n_tickers, brier, mae,
--                         coverage_80, fallback_rate, computed_at.
--                                                      MERGE (model)
-- gold.model_calls        call_id, ts, task, model, latency_ms, ok, in_tokens, out_tokens
--                                                      MERGE (call_id)
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
  task         STRING    NOT NULL COMMENT 'Task name: ingest_prices | ingest_news | build_silver_prices | build_silver_news | build_features | fit_models | run_forecasts | run_backtest. The last three write gold (B-6) through the same ledgered path; run_backtest is on demand, not part of the daily workflow.',
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
  article_url         STRING          COMMENT 'Passed through from bronze.',
  doc_id              STRING          COMMENT 'DERIVED at C-1: article_id || ":" || ticker. PRIMARY KEY OF THE AI SEARCH INDEX, which allows only one key column while this table is grained on (article_id, ticker). Keying the index on article_id alone would let one row of a multi-ticker article win arbitrarily, and the ticker-filtered search would then miss it. Deterministic across re-runs because both inputs are MERGE keys. NOT the table MERGE key, which is unchanged.'
)
USING DELTA
COMMENT 'Normalized news, one row per (article, insight). MERGE on (article_id, ticker) — never a blind INSERT.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- MIGRATION for the already-populated table (C-1). CREATE TABLE IF NOT EXISTS does not alter an
-- existing table, so the column has to be added explicitly. Run this once; it is a no-op
-- afterwards.
--
-- No backfill UPDATE is needed: build_silver_news projects doc_id in its staged SELECT and the
-- MERGE's WHEN MATCHED THEN UPDATE SET * rewrites every matched row, so the next run populates
-- the whole table. (UPDATE SET * / INSERT * match by NAME, which is also why the source SELECT
-- must project doc_id — with schema evolution off, a target column missing from the source fails
-- the INSERT rather than defaulting to NULL.)
--
-- Adding a column to a CDF-enabled table is safe: Change Data Feed handles the schema change, the
-- existing feed is not invalidated, and changes written after this point carry the new column.
ALTER TABLE market_intel.silver.news_articles
  ADD COLUMN IF NOT EXISTS doc_id STRING COMMENT 'DERIVED at C-1: article_id || ":" || ticker. Primary key of the AI Search index.';

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
-- GOLD (B-6) — implemented
-- =====================================================================================
-- Gold is what the app and the agent read. Nothing in it is computed at read time: the modeling
-- layer produces every number in pandas and src/pipelines/write_gold MERGEs it here, so the
-- Streamlit pages and the agent tools are pure readers (the agent NEVER computes statistics).
--
-- SCALES. Every rate in gold is a DECIMAL fraction, never a percent: return_p10 = -0.031 means
-- -3.1%, and the regime means and sigmas match. Estimation happens in percent log returns (B-0)
-- and the divide-by-100 happens before anything is stored, so one display format applies to the
-- whole layer.
--
-- MERGE KEYS ARE THE ROW'S IDENTITY, not its surrogate id. forecast_runs keys on
-- (ticker, as_of_date, model_used) rather than forecast_id: a UUID key can never match an existing
-- row, which would silently turn the required MERGE back into a blind INSERT and duplicate a day's
-- forecast on every retry. forecast_id is instead DERIVED from those same three values (uuid5), so
-- it is stable across re-runs and a saved research report's forecast_id keeps resolving.
--
-- NULL MEANS "NOT APPLICABLE" HERE, and it is used deliberately: prob_low_vol/prob_high_vol are
-- NULL on a GBM forecast because a model without regimes has no regime probability, and
-- backtest_metrics.converged is NULL for GBM because there is no optimizer to converge. Writing
-- 0.0 or true instead would be a claim rather than an absence.

-- The current regime read, one row per ticker per day. This is the "High volatility — 73%" card.
-- prob_low_vol/prob_high_vol are FILTERED probabilities (data through as_of_date only), never
-- smoothed: smoothed probabilities use the whole sample and would make this card a hindsight
-- statement (architecture doc section 5).
CREATE TABLE IF NOT EXISTS market_intel.gold.regime_states (
  ticker              STRING NOT NULL COMMENT 'MERGE key 1.',
  as_of_date          DATE   NOT NULL COMMENT 'Last trade_date included in the fit. MERGE key 2.',
  prob_low_vol        DOUBLE          COMMENT 'Filtered probability of the calm regime at as_of_date. Filtered, never smoothed.',
  prob_high_vol       DOUBLE          COMMENT 'Filtered probability of the turbulent regime. Sums to 1 with prob_low_vol.',
  low_vol_mean        DOUBLE          COMMENT 'Fitted daily mean log return of the calm regime, DECIMAL scale (0.0004 = 0.04%).',
  low_vol_sigma       DOUBLE          COMMENT 'Fitted daily sigma of the calm regime, DECIMAL scale. Always <= high_vol_sigma: regimes are re-sorted by fitted variance after every fit.',
  high_vol_mean       DOUBLE          COMMENT 'Fitted daily mean log return of the turbulent regime, DECIMAL scale.',
  high_vol_sigma      DOUBLE          COMMENT 'Fitted daily sigma of the turbulent regime, DECIMAL scale.',
  current_news_signal DOUBLE          COMMENT 'news_sentiment_3d at as_of_date, the value the forecast decays over the horizon. In [-1, 1].',
  model_used          STRING          COMMENT 'Rung that actually produced this row: news_markov | markov | gbm. Records the C->B->A fallback rather than hiding it.',
  model_version       STRING          COMMENT 'Version of the modeling code, so stored rows stay comparable across changes.'
)
USING DELTA
COMMENT 'Current regime read per ticker. MERGE on (ticker, as_of_date) — never a blind INSERT.';

-- One forecast distribution per ticker per day per model. A DISTRIBUTION, not a point prediction:
-- the quantiles and the probabilities are the product, and the 5,000 raw paths are deliberately
-- NOT stored (B-4).
CREATE TABLE IF NOT EXISTS market_intel.gold.forecast_runs (
  forecast_id       STRING NOT NULL COMMENT 'Stable uuid5 of (ticker, as_of_date, model_used). Referenced by research_reports.forecast_id in Lakebase (C-2), so it must not change on a re-run.',
  ticker            STRING NOT NULL COMMENT 'MERGE key 1.',
  generated_at      TIMESTAMP       COMMENT 'UTC wall-clock when the simulation ran. Not a key: re-running the same day updates this in place.',
  as_of_date        DATE   NOT NULL COMMENT 'Last trade_date the forecast was conditioned on. MERGE key 2.',
  horizon_days      INT             COMMENT 'Forecast horizon in trading days (config forecast.horizon_days = 5).',
  model_used        STRING NOT NULL COMMENT 'news_markov | markov | gbm — the rung that produced this row. MERGE key 3, so all three models can be stored for one day and compared.',
  current_price     DOUBLE          COMMENT 'Close at as_of_date, the price every path starts from.',
  price_p10         DOUBLE          COMMENT '10th percentile simulated price at the horizon.',
  price_p50         DOUBLE          COMMENT 'Median simulated price at the horizon.',
  price_p90         DOUBLE          COMMENT '90th percentile simulated price at the horizon.',
  return_p10        DOUBLE          COMMENT '10th percentile horizon return, DECIMAL (-0.031 = -3.1%). Derived from price_p10 so the two can never disagree.',
  return_p50        DOUBLE          COMMENT 'Median horizon return, DECIMAL.',
  return_p90        DOUBLE          COMMENT '90th percentile horizon return, DECIMAL. [return_p10, return_p90] is the 80% interval the backtest scores coverage against.',
  prob_positive     DOUBLE          COMMENT 'Share of paths with a positive horizon return, i.e. P(R5 > 0). The quantity the Brier score scores.',
  prob_loss_gt_5pct DOUBLE          COMMENT 'Share of paths with a horizon return below -5%, i.e. P(R5 < -0.05), strictly worse than a 5% loss.',
  prob_low_vol      DOUBLE          COMMENT 'Filtered probability of the calm regime the paths were initialized from. NULL for gbm, which has no regimes.',
  prob_high_vol     DOUBLE          COMMENT 'Filtered probability of the turbulent regime. NULL for gbm.',
  n_paths           INT             COMMENT 'Simulated paths (config forecast.n_paths = 5000). Stored because a percentile without its sample size is not interpretable.',
  seed              BIGINT          COMMENT 'Seed of the single numpy Generator used for this run. With model_version, this makes the row reproducible.',
  model_version     STRING          COMMENT 'Version of the simulation code. Bump it when a change makes stored forecasts non-comparable — a different draw order counts.'
)
USING DELTA
COMMENT 'Forecast distributions. MERGE on (ticker, as_of_date, model_used) — never a blind INSERT.';

-- Walk-forward backtest, one row per (origin, ticker, model). Written by the on-demand backtest
-- job, NOT by the daily workflow.
--
-- model vs model_used is the whole fallback story: model is the arm that was asked for, model_used
-- is the rung that answered after the C->B->A descent. The fallback rate the Model Evaluation page
-- shows is simply how often they differ.
--
-- realized_return, return_p50 and prob_positive are stored beyond the spec's column list so a
-- published score can be recomputed from the row that claims it. A Brier score with no record of
-- the probability and the outcome behind it cannot be audited.
CREATE TABLE IF NOT EXISTS market_intel.gold.backtest_metrics (
  origin_date     DATE    NOT NULL COMMENT 'Origin T: the last session in the training window. MERGE key 1.',
  ticker          STRING  NOT NULL COMMENT 'MERGE key 2.',
  model           STRING  NOT NULL COMMENT 'Arm evaluated: news_markov | markov | gbm. All three fit on the IDENTICAL training window at each origin (parity rule). MERGE key 3.',
  brier           DOUBLE           COMMENT 'Squared error of prob_positive against the realized direction. 0 is perfect, 0.25 is an honest coin flip.',
  mae             DOUBLE           COMMENT 'At this grain, the ABSOLUTE error of return_p50 against realized_return. The pooled summary averages these into a mean absolute error.',
  covered_80      BOOLEAN          COMMENT 'Whether realized_return fell inside [return_p10, return_p90], inclusive. Should be true about 80% of the time; over-coverage is as much a failure as under-coverage.',
  model_used      STRING           COMMENT 'Rung that actually produced the forecast. Differs from model exactly when the ladder fell back.',
  converged       BOOLEAN          COMMENT 'Whether the optimizer converged. NULL for gbm, which has no optimizer.',
  failure_reason  STRING           COMMENT 'Recorded reason for every rung that failed above model_used, or NULL. A silent fallback is as bad as a crash.',
  realized_return DOUBLE           COMMENT 'Actual horizon return from origin_date, DECIMAL. Used for scoring only — never for fitting.',
  return_p50      DOUBLE           COMMENT 'Forecast median return, kept so mae can be recomputed from this row.',
  prob_positive   DOUBLE           COMMENT 'Forecast P(R5 > 0), kept so brier can be recomputed from this row.'
)
USING DELTA
COMMENT 'Walk-forward backtest scores. MERGE on (origin_date, ticker, model) — never a blind INSERT.';

-- The pooled summary the Model Evaluation page reads: one row per model, across every origin and
-- ticker of the most recent backtest run.
--
-- A TABLE rather than a view, deliberately. The pooled metrics are already computed in pandas by
-- src/models/backtest.pooled_summary, and a SQL view would be a SECOND implementation of the same
-- three averages — two implementations of one number is how a page and its evidence start
-- disagreeing. So the aggregation exists once, in tested Python, and this table stores its output.
--
-- n IS MANDATORY ON THE PAGE. 26 weekly origins across 5 tickers is 130 forecasts per model, which
-- is a small sample for a Brier difference. "No meaningful improvement detected at this sample
-- size" is a first-class verdict (spec A2), and it cannot be stated honestly without n.
CREATE TABLE IF NOT EXISTS market_intel.gold.backtest_summary (
  model         STRING NOT NULL COMMENT 'news_markov | markov | gbm. MERGE key: one current pooled row per model, replaced by the next backtest run.',
  n             BIGINT          COMMENT 'Scored (origin, ticker) forecasts behind these numbers. ALWAYS displayed on the Model Evaluation page.',
  n_tickers     INT             COMMENT 'Distinct tickers pooled, so a large n from one ticker is distinguishable from a broad one.',
  brier         DOUBLE          COMMENT 'Mean Brier score over the n forecasts. Lower is better.',
  mae           DOUBLE          COMMENT 'Mean absolute error of the median return, DECIMAL.',
  coverage_80   DOUBLE          COMMENT 'Share of outcomes inside the 80% interval. The target is 0.80, from either direction.',
  fallback_rate DOUBLE          COMMENT 'Share of origins where this arm fell back to a simpler rung. 0 for gbm, which is the bottom of the ladder. Displayed alongside n.',
  computed_at   TIMESTAMP       COMMENT 'UTC wall-clock of the backtest run that produced this row, so the page can say how stale the verdict is.'
)
USING DELTA
COMMENT 'Pooled backtest summary, one row per model. MERGE on (model) — never a blind INSERT.';

-- =====================================================================================
-- GOLD (C-3) — implemented
-- =====================================================================================
-- One row per model call, buffered in memory by src/llm/telemetry.py and MERGEd here on flush.
-- Instrumentation only: no routing subsystem reads it, and nothing in the product depends on it.
--
-- call_id EXISTS SO THE WRITE CAN BE A MERGE. The spec's column list ({ts, task, model,
-- latency_ms, ok, in_tokens, out_tokens}) has no identity, and rule 4 forbids a blind INSERT, so
-- every record carries a uuid4 and that is the key. Without it a retried flush — the app's flush
-- is best-effort and re-queues its records when a write fails — would duplicate rows.
--
-- FAILED CALLS ARE ROWS TOO: ok = false with NULL token counts. A telemetry table holding only
-- successes cannot answer the question it exists for.
CREATE TABLE IF NOT EXISTS market_intel.gold.model_calls (
  call_id    STRING    NOT NULL COMMENT 'uuid4 minted when the record was buffered. MERGE key — the record has no natural identity, and a MERGE needs one.',
  ts         TIMESTAMP          COMMENT 'UTC wall-clock when the call was recorded (call completion, success or failure).',
  task       STRING             COMMENT 'The call_model task, i.e. which config endpoint was used: agent | slm. NOT a model tier — there is no router (architecture doc section 17).',
  model      STRING             COMMENT 'Model the endpoint reported, falling back to the serving endpoint name when the response omits it.',
  latency_ms DOUBLE             COMMENT 'Wall-clock milliseconds around the HTTP call, including any retry waits — what the user actually waited.',
  ok         BOOLEAN            COMMENT 'Whether the call returned a usable completion. False rows carry the latency of the failure and NULL tokens.',
  in_tokens  BIGINT             COMMENT 'usage.prompt_tokens when the endpoint reports it, else NULL. NULL means "not reported", never 0.',
  out_tokens BIGINT             COMMENT 'usage.completion_tokens when the endpoint reports it, else NULL.'
)
USING DELTA
COMMENT 'LLM call telemetry (C-3). MERGE on (call_id) — never a blind INSERT.';
