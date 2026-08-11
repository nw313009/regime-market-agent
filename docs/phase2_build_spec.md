# Phase 2 — Build Specification
## Regime-Aware Market Intelligence Agent (frozen v2.1)

This document is written to be given to Cursor as project context, alongside the frozen
architecture doc. It contains: (Part A) how the system behaves functionally, (Part B) how
it is built, checkpoint by checkpoint, with contracts, schemas, and tests, and (Part C)
the Databricks deployment workflow — the part code generation cannot infer.

RULES FOR CURSOR (paste these into project rules):
1. The architecture is frozen (v2.1). Section 17 of the architecture doc lists banned
   components. Do not introduce them. Do not "improve" the architecture.
2. Implement exactly the contracts in this spec. Where the spec is silent, choose the
   simplest working option and leave a TODO comment.
3. All statistical constraints in Section 5 of the architecture doc are mandatory:
   left-stochastic transition matrix handling, lagged exog_tvtp, filtered probabilities
   only, per-fit regime re-sorting, C→B→A fallback ladder.
4. Spark handles ingestion → silver → features. Modeling runs in pandas/statsmodels
   after a .toPandas() at the daily_features boundary. Do not write the models in Spark.
5. No secrets in code. API keys come from Databricks secrets / environment.
6. Every pipeline write is idempotent (MERGE on declared keys, never blind INSERT).

---

# PART A — FUNCTIONAL WALKTHROUGH

## A1. What happens every day (the workflow, task by task)

Scheduled once daily after US market close (suggest 22:30 UTC), plus manual "Run now".

1. ingest_prices    — Massive aggregates for each ticker in universe (seed 5 + all
                      watchlist tickers) since last stored trade_date → MERGE into
                      bronze.prices_raw.
2. ingest_news      — Massive news per ticker since last stored published_at → MERGE
                      into bronze.news_raw.
3. build_silver     — Bronze → silver.daily_prices and silver.news_articles (explode
                      multi-ticker articles; normalize sentiment; build embedding_text).
4. build_features   — silver → silver.daily_features (returns, rolling vol, volume z,
                      news sentiment with weekend roll-forward and 3-day decay).
5. fit_models       — per ticker: fit C (fallback B, fallback A) on full available
                      history; extract sorted regime params + filtered current-regime
                      probabilities → gold.regime_states.
6. run_forecasts    — per ticker: 5,000-path × 5-day Monte Carlo from current price,
                      current filtered regime distribution, sorted params, decayed news
                      → one row in gold.forecast_runs.
7. sync_news_index  — trigger AI Search index sync (reads Delta CDF of news_articles).

The walk-forward backtest is NOT part of the daily job. It is a separate job/notebook
run on demand; results land in gold.backtest_metrics and the app reads them.

## A2. The three pages, as the user experiences them

MARKET RESEARCH — select ticker → price history chart; current regime card
("High volatility — 73%", from gold.regime_states filtered probs); forecast distribution
(P10/P50/P90 of 5-day return, P(positive), P(loss>5%)); recent news list with sentiment
and news_count context ("no relevant news" vs "neutral news"); the decay-assumption
disclosure sentence.

RESEARCH AGENT — chat box scoped to a ticker. User asks e.g. "why is downside risk
elevated this week?" Agent turn-by-turn:
  1. calls get_market_forecast(ticker) → structured Gold numbers
  2. calls search_market_news(ticker, query derived from question) → top-k articles
  3. writes an explanation that cites the numbers and the retrieved articles, and
     invents neither
  4. buttons / commands: "Save this as a report" → save_research_report;
     "add NVDA to my watchlist" → update_watchlist
The agent NEVER computes statistics. It reads Gold and explains.

MODEL EVALUATION — pooled table: GBM vs Markov vs News-Markov on Brier, median-return
MAE, 80% interval coverage; n displayed on the page; fallback rate displayed (how often
C failed to converge); an honest verdict line that is generated from the numbers, with
"no meaningful improvement detected at this sample size" as a first-class outcome.

## A3. The CDC demo moment (rehearse this exact sequence)

1. In the agent page: "Add AMD to my watchlist." → tool call → row appears in
   Lakebase watchlist_tickers (show it via a query).
2. Show the same change arriving in the Delta history table (lb_watchlist_tickers
   history) via Lakebase CDF.
3. One sentence: "Operational writes in Postgres, captured through the WAL into the
   lakehouse — that's the CDC path; the analytical data flows the other way and the
   two never loop."

## A4. Full demo script (the 13 steps, condensed)

Run pipeline → show Bronze/Silver → show daily_features → show current regime →
show forecast distribution → show evaluation table with n → ask agent why risk
changed → agent retrieves news + explains → add ticker to watchlist → save report →
show Lakebase rows → show CDC arrival in Delta → stop.

---

# PART B — TECHNICAL BUILD SPEC BY CHECKPOINT

## B0. Repository layout (frozen from v1.0, minor additions)

regime-market-agent/
├── app/                    # Streamlit Databricks App
│   ├── app.py
│   ├── app.yaml
│   ├── requirements.txt    # app-only deps, consumed by the Databricks App
│   └── pages/ market_research.py | research_agent.py | model_evaluation.py
├── src/
│   ├── ingestion/  massive_client.py | ingest_prices.py | ingest_news.py
│   ├── pipelines/  silver_prices.py | silver_news.py | feature_pipeline.py
│   ├── models/     gbm.py | markov.py | news_markov.py | monte_carlo.py | backtest.py
│   ├── agent/      agent.py | tools.py | prompts.py
│   ├── llm/        call_model.py | telemetry.py
│   └── database/   delta.py | lakebase.py
├── notebooks/      00_smoke_test.py | 10_backtest_run.py   # thin wrappers only
├── setup/          create_catalog.sql | create_delta_tables.sql |
│                   create_lakebase.sql | create_ai_search.py |
│                   create_workflow.py | seed_demo_data.py
├── sql/            (intentionally EMPTY — see sql/README.md)
│                   # v1.0 planned this as a reviewable mirror of setup/'s DDL. Dropped at B-6:
│                   # a hand-copied second definition with no test keeping it honest drifts, and
│                   # a drifted DDL reference is worse than none — it is the file a reader trusts
│                   # while the cluster runs the other one. setup/create_delta_tables.sql is the
│                   # single source and is written to be read (a COMMENT on every column); the
│                   # tests parse it and assert the columns, NOT NULL MERGE keys and ledger task
│                   # names match the code, which a mirror could not do. Section 15 (environment
│                   # recreatable from code) is satisfied by setup/ alone.
├── tests/          test_features.py | test_models.py | test_monte_carlo.py |
│                   test_no_lookahead.py | test_agent_tools.py | test_idempotency.py |
│                   test_massive_client.py | test_ingestion.py | test_silver.py |
│                   test_call_model.py | test_telemetry.py | test_lakebase.py |
│                   test_agent_loop.py | test_ai_search.py |
│                   test_import_boundaries.py | conftest.py
│                   # test_massive_client/test_ingestion/test_silver/conftest added at A-1/A-2/A-3:
│                   # the vendor client, the ingestion watermark/row-building logic and the silver
│                   # derivations had no home in the original list, and conftest.py holds the
│                   # payload fixtures they share. test_call_model/test_telemetry/test_lakebase
│                   # added at C-2/C-3 for the same reason: C-7 names only test_agent_tools, which
│                   # is the agent's integration test, and the model wrapper, the telemetry buffer
│                   # and the Postgres access layer are all unit-testable without a workspace.
│                   # test_agent_loop/test_ai_search/test_import_boundaries added at C-1/C-4: the
│                   # loop's iteration cap and the setup script's idempotency are exactly the
│                   # properties an integration test cannot demonstrate cheaply (you would have to
│                   # provoke a runaway model and re-create a populated index), and the psycopg
│                   # import boundary is a static property of the source tree, not of a run.
├── config/config.yaml
├── requirements.txt              # local 3.12 venv, fully pinned (dev + pytest)
└── requirements-databricks.txt   # ONLY packages the Databricks runtime lacks:
                                  # statsmodels, exchange_calendars,
                                  # databricks-sql-connector
                                  # (+ databricks-sdk as a version floor, not an absence)
                                  # psycopg REMOVED at C-1 — see the environment note below

Dependency environments are split three ways, recorded here under rule 2 (spec silent →
simplest working option). requirements.txt fully pins the local Python 3.12 dev/test venv —
statsmodels, pandas, numpy, exchange_calendars, databricks-sql-connector,
psycopg[binary,pool], databricks-sdk, streamlit, requests, pyyaml, pytest — with
statsmodels==0.14.6 as a hard floor, since earlier versions fail to import under pandas 3.0;
it is never installed on a cluster. requirements-databricks.txt is what notebooks and job
environments install, and lists only what the runtime lacks (statsmodels,
exchange_calendars, databricks-sql-connector) plus databricks-sdk,
which the runtime does ship but often below the >=0.125.0 floor that
w.postgres.generate_database_credential requires: NEVER pip-upgrade pandas, numpy or pyarrow
on a cluster, because the runtime pins those three against its own Spark build and replacing
them breaks Spark in ways that surface far from the change. app/requirements.txt carries what
the Databricks App consumes (streamlit, databricks-sql-connector, psycopg[binary,pool],
databricks-sdk, pyyaml, requests); the app has no SparkSession and fits no models, so it needs
no statsmodels. Lakebase uses psycopg v3 rather than psycopg2 because the proven connection
pattern (C-2) depends on psycopg_pool.ConnectionPool with per-connection OAuth credentials.
The frozen architecture doc is silent on dependency files and is unaffected by this split.

ENVIRONMENT NOTE (C-1): psycopg KILLS THE SERVERLESS NOTEBOOK KERNEL, so Lakebase is
app-container-only.

Symptom, observed: `import psycopg` (psycopg[binary] 3.3.4) on serverless compute aborts the
kernel at import time. The libpq extension abort fires inside psycopg/pq/__init__.py
import_from_libpq and the process exits 134 (SIGABRT). It is not an exception — no traceback,
no try/except, nothing of ours runs. The same package works in the Databricks App container,
where the pooled-OAuth pattern is proven, and in the local venv.

Boundary, enforced: psycopg is removed from requirements-databricks.txt; src/database/lakebase.py
is imported only by the app and by src/agent/tools.py's two WRITE tools, which import it inside
the function rather than at module level; nothing under src/ingestion, src/pipelines or
src/models may reach it through any chain of first-party imports. tests/test_import_boundaries.py
walks the import graph and asserts all of that, including that `import src.agent.tools` (for the
tool schemas, or from a notebook) pulls in no psycopg.

This costs nothing architecturally: Lakebase is authoritative only for application state and the
pipelines deal exclusively in Delta. The one operational consequence is that ensure_tables()
cannot be called from a notebook, so the market_system schema was created by running
setup/create_lakebase.sql in the SQL editor — all four tables exist and REPLICA IDENTITY FULL is
verified on watchlist_tickers and research_reports. ensure_tables() stays for the app's startup
path and as the executable record of the DDL.

config.yaml keys: massive.base_url, massive.rate_limit_per_min,
massive.backfill_start_date="2024-08-01", tickers.seed[5],
forecast.horizon_days=5, forecast.n_paths=5000, forecast.seed=42,
news.half_life_days=2, backtest.min_train_days=252, backtest.origin_freq=weekly,
backtest.n_weeks=26, model.agent_endpoint, model.slm_endpoint (unused until stretch),
catalog=market_intel, telemetry.mode (added at C-3), search.* (added at C-1).

search.* was added at C-1 because the original key list names the index but not the knobs around
it: search.endpoint_name, search.index, search.source_table, search.primary_key,
search.embedding_source_column, search.embedding_model_endpoint, search.top_k. Table names are
stored catalog-relative and prefixed with `catalog` at the edge, exactly as in the pipelines, so
pointing the project at a second catalog stays a one-line change. Two modules read the section —
setup/create_ai_search.py to build the index and src/agent/tools.py to query it — and
tests/test_agent_tools.py asserts they resolve the same index name, since drift there would have
the agent querying an index nothing creates.

telemetry.mode was added at C-3 because the original key list predates the question of where
model-call telemetry goes, and there is no single answer: the spec's destination is a Delta
table, and the Databricks App that makes most of the calls has no SparkSession (C-5). Values are
delta | log | off; the file says delta (notebooks and jobs) and app.yaml overrides it with the
TELEMETRY_MODE environment variable. One config, one documented exception, rather than two
configs that drift.

massive.backfill_start_date was added at A-2 (the original key list had no backfill window, and
the ingestion tasks need one for the empty-table case). It is a FIXED ISO date rather than a
rolling lookback because a rolling window fetches a different dataset on every run, which breaks
reproducibility — setup/ plus this config should recreate the same Bronze — and silently shifts
the backtest sample between runs. Empty table → fetch from this date; populated table → fetch
from the per-ticker watermark.

## CHECKPOINT A — Data works

### A-0 Smoke test (FIRST ACTION, before anything else)
Implemented as notebooks/00_smoke_test.py — the SINGLE smoke test. No local duplicate script
(B0 lists one file; a near-duplicate drifts and then disagrees with itself).

Dual auth, no secret ever in the file:
  local          MASSIVE_API_KEY from the environment, falling back to repo-root .env
  workspace cell getpass prompt (typed, not stored)
  scheduled job  dbutils.secrets.get(scope="capstone", key="massive_api_key")
Secret setup: databricks secrets create-scope capstone; databricks secrets put-secret
capstone massive_api_key.

Both routes are VERIFIED live (200, same session) — treat both as confirmed and do NOT
re-derive these paths from memory:
  aggregates  /v2/aggs/ticker/NVDA/range/1/day/<from>/<to>  → 22-bar results payload
  news        /v2/reference/news?ticker=NVDA&limit=N        → {count, next_url, request_id,
                                                              results, status}

OUTPUT RULE: print the status code, the request_id, and structural facts about the payload
(key names, counts) ONLY. NEVER print a response body or a full URL. apiKey travels as a
query parameter and error payloads can echo it into logs, notebook output, or an agent
transcript — the r.text[:500] form this spec previously showed is banned for that reason
(A-1 security requirement).

200 → proceed to A-1. Anything else → stop and diagnose (401/403 = key or plan; connection
error = egress). Note the specific case: "API Key was not provided" means the key never
reached the request at all, so check env/.env loading before suspecting the key or the plan.

### A-1 massive_client.py contract

    class MassiveClient:
        def __init__(cfg, secret_getter): ...
        def get_daily_aggregates(ticker, start_date, end_date) -> list[dict]
        def get_news(ticker, published_after) -> list[dict]

Requirements: token-bucket or simple sleep-based throttle from
cfg.rate_limit_per_min; follow pagination to exhaustion via next_url from the response
envelope (VERIFIED live — the envelope is {count, next_url, request_id, results, status});
retry on 429/5xx with exponential backoff + jitter (max 5 attempts); raise on 401/403 with a
clear message; log every request (url minus key, status, latency).

SECURITY — apiKey travels as a QUERY PARAMETER, so URLs and error bodies are credential-
bearing. On a non-200, NEVER log or print the response body or the full URL. Log only: the
status code, the request_id if it parses out of the payload, and the endpoint name (e.g.
"reference/news"). This is why a 401 body must not be echoed: Massive's error payloads and
redirected URLs can reflect request params straight into your logs, notebook output, or an
agent transcript. Same rule in the A-0 smoke test.

### A-2 Bronze
bronze.prices_raw, bronze.news_raw: store the near-raw payload rows plus
source, ingested_at, request_id, ticker, source_timestamp. bronze.ingestion_runs:
run_id, task, started_at, finished_at, status, rows_written, error. MERGE keys:
prices (ticker, source_timestamp); news (article_id, ticker) after a light explode.

The news explode is FROM insights, not from the tickers array — same rule as A-3, applied one
layer earlier so the bronze row already carries the per-insight ticker alongside its raw
sentiment/sentiment_reasoning. Exploding tickers here would manufacture bronze rows with no
sentiment and would put the A-3 rule one refactor away from regressing. Consequence, accepted:
a ticker listed in tickers with no insight produces no bronze row.

Bronze prices key on source_timestamp (the epoch-ms bar start, resolved in the exchange
timezone) rather than trade_date, which is silver's grain — bronze stays near-raw and the
session-date mapping happens once, in silver.

### A-3 Silver
(News shapes below are VERIFIED against a live Massive response, not inferred.)

silver.daily_prices — ticker, trade_date, open, high, low, close, volume, vwap.
MERGE on (ticker, trade_date).

Timestamp parsing is PER-SOURCE. Aggregates: t is epoch-milliseconds. News:
published_utc is an ISO-8601 UTC string (e.g. "2026-08-10T02:15:00Z"). Both then map to
the trading session in exchange timezone (America/New_York), never UTC-naive. Only the
parsing step differs; the session rule does not.

silver.news_articles — article_id, ticker, published_at, title, description,
publisher, sentiment_label, sentiment_score, sentiment_reasoning,
embedding_text (= title + "\n" + description), article_url, doc_id (added at C-1).
MERGE on the composite key (article_id, ticker). CDF at creation:
  TBLPROPERTIES (delta.enableChangeDataFeed = true)

doc_id = article_id || ":" || ticker, derived in the silver build. It exists for one reason: an
AI Search Delta Sync index takes exactly ONE primary key column and this table is grained on
(article_id, ticker). Keying the index on article_id alone would let one row of a multi-ticker
article win arbitrarily, so search_market_news("MSFT", ...) would answer "no relevant news" for
an article that exists — a silent retrieval hole rather than a visible error. Deterministic
across re-runs, because both inputs are the MERGE key. The MERGE key itself is UNCHANGED.

Accepted cost: the embedding is computed per ROW, so a 3-ticker article embeds 3 times. That is
already the shape of the insights explode, and at roughly 28k rows it is bounded and small.

Migration, since the table was already populated when the column was added: CREATE TABLE IF NOT
EXISTS does not alter an existing table, so setup/create_delta_tables.sql carries an ALTER TABLE
... ADD COLUMN IF NOT EXISTS doc_id after the CREATE. No backfill UPDATE is needed — the build's
staged SELECT projects doc_id and the MERGE's WHEN MATCHED THEN UPDATE SET * rewrites every
matched row, so the next run populates the whole table. (UPDATE SET * and INSERT * match by NAME,
which is also why the source SELECT must project the column: with schema evolution off, a target
column missing from the source fails the INSERT rather than defaulting to NULL.) Adding a column
to a CDF-enabled table is safe: the existing feed is not invalidated and changes written
afterwards carry the new column.

EXPLODE FROM insights, NOT tickers. Each article carries both a tickers array and an
insights array of {ticker, sentiment, sentiment_reasoning}. Sentiment exists only inside
insights, so the explode is one row per (article, insight), taking BOTH ticker and
sentiment from the insight. Deliberate consequence, accepted: a ticker listed in
tickers with no corresponding insight produces NO row. Exploding tickers instead would
manufacture rows with null sentiment. Observed live: insights tickers are a subset of
tickers (10/10 articles, zero mismatches), so treat strict-subset as normal, not an anomaly.

Field mapping from the Massive payload:
  article_id          ← id                  (stable 64-char hex digest)
  published_at        ← published_utc       (ISO-8601 UTC, parse then map to session)
  publisher           ← publisher.name      (publisher is a nested dict:
                                             name/homepage_url/logo_url/favicon_url)
  ticker              ← insights[].ticker
  sentiment_label     ← insights[].sentiment           (raw: positive/neutral/negative)
  sentiment_reasoning ← insights[].sentiment_reasoning (raw text, kept for agent/UI display)
  title, description, article_url pass through unchanged.

sentiment_score is DERIVED at silver build time, not ingested. Massive returns NO numeric
score — insight keys are exactly {ticker, sentiment, sentiment_reasoning}. Map
positive→+1, neutral→0, negative→−1; any unrecognized label→0 AND log a warning, so a new
vendor label degrades to neutral instead of failing or being silently dropped.
daily_features s_t consumes sentiment_score unchanged (A-4).

### A-4 feature_pipeline.py → silver.daily_features
Grain: (ticker, trade_date), trading days only (exchange_calendars, calendar XNYS).

Columns and definitions:
  log_return        = ln(close / lag(close))                 [window: ticker, by date]
  return_5d         = close / lag(close, 5) − 1
  momentum_5d       = sum of last 5 log_returns
  realized_vol_20d  = stddev_samp(log_return) over trailing 20 rows
  volume_zscore_20d = (volume − mean_20) / stddev_20
  news assignment   : map published_at → its trading session; if market closed,
                      NEXT session from the calendar (never weekday arithmetic)
  s_t               = mean of silver.news_articles.sentiment_score for that session
                      (already ±1/0 — derived at silver build, see A-3); 0 if no articles
  news_sentiment_3d = (1.0·s_t + 0.5·s_{t−1} + 0.25·s_{t−2}) / 1.75
  news_count        = article count mapped to that session
Spark window functions partitioned by ticker ordered by trade_date. Rows with null
rolling features (warm-up period) stay in the table; the modeling layer drops them.

Decisions taken where the above is silent (A-4 implementation):
- close and volume are CARRIED into daily_features from daily_prices. B-4 reads the last close
  as current_price and A-4 needs volume for the z-score, and C-b makes daily_features the only
  table the modeling layer reads — so requiring a second table read there would break that line.
- Every rolling column is NULL until its frame is FULL, enforced with a count over the same
  frame. Spark aggregates skip NULLs, so a 20-row frame holding 19 usable returns would
  otherwise report a 19-observation stddev as a 20-day one. First non-null: momentum_5d row 5,
  realized_vol_20d row 20, volume_zscore_20d row 19 (one earlier — volume has no undefined
  first row).
- mean_20/stddev_20 for volume_zscore_20d use the trailing 20 rows INCLUDING the current row,
  the same frame as realized_vol_20d, and stddev_20 is stddev_samp for consistency with it. A
  constant window (zero deviation) yields NULL, not an infinity, via try_divide.
- news_sentiment_3d treats a missing s_{t−1}/s_{t−2} at the start of history as 0, since the
  denominator is fixed at 1.75 and "no articles" is already defined as 0. The column is never
  NULL.
- The date → session lookup is built on the driver from the XNYS calendar and broadcast to
  Spark as a small temp view, so the calendar is consulted in exactly one place and the join is
  a plain equality join. Its range is the price date range: an article rolling past the last bar
  is picked up by the run that ingests that bar.
- The bronze.ingestion_runs ledger (A-2) is extended to the pipeline tasks. build_silver_prices,
  build_silver_news and build_features each write one row per call, same shape, on success and
  on failure, through the shared write layer — so a workflow run leaves one audit row per task.

### A-5 Checkpoint A tests
test_features: known synthetic price series → exact expected log_return, vol, z.
test_weekend_news: Saturday+Sunday articles land in Monday's s_t and news_count.
test_idempotency: run silver build twice on same Bronze → identical row counts and
checksums.

Implemented as: exact values on series with closed forms (constant log returns via a geometric
close series, alternating ±r for a non-degenerate stddev, volumes [100]×19+[200] for a z-score
of 95/√500), the warm-up boundaries above, and the calendar cases. test_weekend_news is
accompanied by WEEKDAY-HOLIDAY cases — Good Friday 2026-04-03 and the observed Independence Day
2026-07-03 are both Fridays, so weekday arithmetic assigns their articles to the same day while
the calendar assigns them to the following Monday. A weekday-arithmetic implementation passes
the weekend test and fails these, which is why both exist. test_idempotency needs Delta and
stays on the workspace integration list.

FIXTURES must mirror the REAL payload shapes for both endpoints — no invented schemas.
Fixtures are the only place the vendor contract is pinned in code, so a fixture that
disagrees with production is worse than no fixture: it makes wrong code pass.
Required for news: nested publisher dict (assert the mapping takes publisher.name), id as a
64-hex digest, published_utc as an ISO-8601 UTC string, and insights entries carrying exactly
{ticker, sentiment, sentiment_reasoning} with NO numeric score. Include at least one article
where insights is a STRICT subset of tickers, and assert the extra ticker yields no row —
that is the A-3 explode rule, and it is the one that silently regresses if someone
"simplifies" the explode back to the tickers array. Include one article with an unrecognized
sentiment label and assert sentiment_score → 0 plus a logged warning.

## CHECKPOINT B — Math works
(The largest and most failure-prone checkpoint. Everything here runs in pandas after
df = spark.table("...daily_features").filter(ticker).toPandas().)

### B-0 Numerical conventions (tell Cursor explicitly)
Work in PERCENT log returns for estimation: r_pct = 100 × log_return. Markov-switching
MLE converges far more reliably at that scale; divide fitted μ, σ by 100 before
simulation. Drop warm-up NaNs. Refuse to fit with < min_train_days observations.

### B-1 gbm.py
fit_gbm(returns) -> {mu, sigma} over the SAME training window the Markov models use
at that origin (parity rule). simulate handled by monte_carlo.py.

### B-2 markov.py

    def fit_markov(returns_pct, exog_tvtp=None) -> FitResult | raises FitError
    def sort_regimes(res) -> SortedParams
    # SortedParams: mus[2], sigmas[2] (low-vol first), P (2x2, LEFT-stochastic,
    #   sorted), filtered_current (2,), perm, converged, degenerate_flags

fit_markov: MarkovRegression(endog, k_regimes=2, trend="c",
switching_variance=True, exog_tvtp=...); fit with search_reps≈20 (multiple random
starts); wrap in try/except; treat non-finite params or optimizer failure as FitError.

Degeneracy checks (treat as FitError for ladder purposes):
  sigmas ratio |σ1/σ0 − 1| < 0.05  → regimes indistinguishable
  any diagonal transition prob > 0.995 or < 0.005 → absorbing/degenerate

sort_regimes: perm = argsort(fitted sigmas); apply perm to mus, sigmas; transition
matrix: P_sorted = P[np.ix_(perm, perm)] (valid for left-stochastic — permutes next-
state rows and prev-state columns consistently); filtered probs: reorder columns of
res.filtered_marginal_probabilities by perm; take LAST ROW as filtered_current.
NEVER read smoothed_marginal_probabilities anywhere in src/models/.

### B-3 news_markov.py
exog_tvtp construction (THE alignment rule):

    n = df["news_sentiment_3d"].shift(1)          # lag one trading day
    exog_tvtp = np.column_stack([np.ones(len(n)), n])
    # drop the first row jointly with endog so lengths match

statsmodels uses exog_tvtp row t to build the transition INTO t; shifting by one day
makes "news known at t affects t→t+1" true. Column of ones is mandatory (intercept).

### B-4 monte_carlo.py

    def run_forecast(sorted_params, model_res_or_none, current_price,
                     current_news, cfg, rng) -> ForecastSummary

Per path (5,000), per day h in 1..5:
  news_h = current_news * exp(−ln(2)/half_life * h)
  if TVTP model: P_h = res.model.regime_transition_matrix(
        res.params, exog_tvtp=np.array([[1.0, news_h]]))[:, :, 0]
        then apply the SAME perm as sort_regimes to P_h
     else: P_h = sorted static P
  next_regime ~ Categorical(P_h[:, current_regime])     # COLUMN = prev state
  r ~ Normal(mus[next_regime], sigmas[next_regime])     # decimal scale
  price *= exp(r); current_regime = next_regime
Initial regime per path ~ Categorical(filtered_current).
GBM paths: r ~ Normal(mu, sigma), no regimes.
rng = np.random.default_rng(cfg.seed) — one Generator per forecast run; store seed.
Outputs: price/return P10, P50, P90; prob_positive = mean(R5 > 0);
prob_loss_gt_5pct; regime probabilities; n_paths; model_version. Do NOT persist raw
paths.

### B-5 backtest.py
Origins: weekly, last cfg.n_weeks weeks, per ticker, each with ≥ min_train_days
training rows ending at T. At each origin: build exog through T only; fit ladder
C → B → A (record model_used and failure reason); forecast T+5 via monte_carlo with
news decay from N_T; score against realized 5-day return.
Metrics (pooled across tickers, with n reported): Brier on P(R5>0); MAE of median
return; 80% interval coverage. Also per-model fallback rate. Write one row per
(origin, ticker, model) to gold.backtest_metrics plus a pooled summary table.
Parity rule: at each origin all three models fit on the identical training window.

### B-6 Gold DDL
gold.regime_states — ticker, as_of_date, prob_low_vol, prob_high_vol, low/high
mean+sigma, current_news_signal, model_used, model_version.
gold.forecast_runs — forecast_id, ticker, generated_at, as_of_date, horizon_days,
model_used, current_price, price_p10/50/90, return_p10/50/90, prob_positive,
prob_loss_gt_5pct, prob_low_vol, prob_high_vol, n_paths, seed, model_version.
gold.backtest_metrics — origin_date, ticker, model, brier, mae, covered_80 (bool),
model_used, converged, plus a pooled_summary view/table with n.

### B-7 Checkpoint B tests (all mandatory before freeze of B)
test_log_returns — exact values on synthetic prices.
test_transition_orientation — for fitted P: assert np.allclose(P.sum(axis=0), 1).
test_stationary — solve π from P (left eigenvector, column convention); simulate
  200k steps with the sampler; empirical frequencies within 1% of π. Catches a
  transposed sampler instantly.
test_tvtp_no_lookahead — fit + forecast at origin T; then corrupt all
  news_sentiment_3d after T with random values; refit/forecast; assert bit-identical
  ForecastSummary. Also assert grep-level: no reference to
  smoothed_marginal_probabilities under src/models/.
  The grep scan uses rglob, not glob (changed at C-1 after Gate 2 review): a non-recursive scan
  silently exempts any future subpackage under src/models/, and "the leak test passed because it
  never looked" is the worst way for this check to fail. The pyspark-boundary scan in
  test_models.py was fixed the same way.
test_mc_seed — same seed → identical percentiles; different seed → different.
test_fallback — inject a FitError from C; assert B used and recorded.

## CHECKPOINT C — Product works

### C-1 AI Search
setup/create_ai_search.py: create endpoint (one, STANDARD); create Delta Sync index
market_intel.silver.news_index on silver.news_articles, embedding source column
embedding_text (managed embeddings), sync mode TRIGGERED. sync_news_index task in the
workflow triggers it. Query path: hybrid search, filter by ticker, top_k≈5.

Decisions taken where the above is silent (C-1 implementation):
- PRIMARY KEY is the derived doc_id (article_id:ticker), not article_id. An index takes one key
  column and this table has a composite grain; see the A-3 note for the retrieval hole that
  article_id would open and the migration the new column needed.
- IDEMPOTENT BY ASKING FIRST. ensure_endpoint and ensure_index look the resource up and return
  the existing one untouched, catching ResourceAlreadyExists for the overlapping-run case. The
  index is NEVER re-created: dropping a populated one discards every embedding and re-embeds the
  table. A misconfigured index has to be deleted deliberately, by hand.
- index_subtype=HYBRID, matching the query path. The SDK documents VECTOR as unsupported.
- columns_to_sync is explicit rather than "all columns": it is exactly what search_market_news
  returns, so description and sentiment_reasoning (both long) stay out of the index and the
  snippet is cut from embedding_text, which already begins with the title. test_ai_search.py
  asserts every synced column exists in the silver projection, because a typo here fails at sync
  time in the workspace rather than at edit time.
- wait_until_ready is hand-rolled with a deadline: the SDK ships a waiter for the endpoint but
  not for the index, and a wait with no deadline is how a notebook hangs overnight. Readiness is
  the property that matters — an index that exists but is not ready errors on query rather than
  returning zero results.
- SDK surface was verified against the installed databricks-sdk 0.125.0 rather than recalled:
  w.vector_search_endpoints (get_endpoint / create_endpoint / the ONLINE waiter) and
  w.vector_search_indexes (get_index / create_index / sync_index / query_index).

### C-2 Lakebase
INSTANCE. The capstone gets its OWN Lakebase project: regime-market-database, Autoscaling
capacity, Postgres branch production (endpoint primary). It does NOT share the instance that
hosts ticket_system and weather_system.

SCHEMA. Its tables nonetheless live in their own schema, market_system, and every query fully
qualifies it — market_system.watchlist_tickers, never bare watchlist_tickers. A dedicated
project removes the collision risk but not the reason to qualify: search_path is not a
contract, qualification keeps grants and CDF targets unambiguous, and it keeps the convention
identical to ticket_system and weather_system so all three projects read the same way.

setup/create_lakebase.sql: creates schema market_system, then
market_system.users(user_id, display_name, created_at);
market_system.watchlists(watchlist_id, user_id, name, created_at);
market_system.watchlist_tickers(watchlist_id, ticker, added_at, added_by, PRIMARY KEY
(watchlist_id, ticker)); market_system.research_reports(report_id, user_id, ticker,
question, report_md, forecast_id, created_at).

CONNECTION — the proven pattern, do not reinvent it. db.py is copied in from a prior
project and is known to work against this Lakebase instance. src/database/lakebase.py
WRAPS db.py; it does not replace, rewrite, or "improve" it. Adapt around its interface.
The pattern it implements:
  - psycopg v3 (psycopg[binary,pool]), NOT psycopg2.
  - psycopg_pool.ConnectionPool holds the connections.
  - Per-connection OAuth: each connection authenticates with a short-lived credential from
    w.postgres.generate_database_credential (databricks-sdk >= 0.125.0). There is no static
    password anywhere, which is what keeps rule 5 true.
  - max_lifetime=3000 — connections are recycled before the OAuth credential expires.
    A pool that outlives its token fails intermittently under load, which is the worst way
    to discover this.
  - check=ConnectionPool.check_connection — dead connections are detected on checkout
    rather than surfacing as a query error in the app.

src/database/lakebase.py exposes small functions over that pool: add_ticker, remove_ticker,
save_report, get_watchlist. Parameterized SQL only, fully qualified table names only.

SERVICE PRINCIPAL — known gotcha, handle before first deploy. A new Databricks App gets a
new service principal, and that identity does not exist in Postgres yet. Before the app can
read or write anything it needs (a) a Postgres role created for it via the
regime-market-database project's OAuth tab, and (b) explicit grants on schema market_system
and its tables. Skipping this
produces an authentication or permission error at first deploy that looks like broken
application code and is not. Do this as part of deployment, not as debugging.

Enable Lakebase CDF on market_system.watchlist_tickers and market_system.research_reports →
Delta history tables (per current docs; if the preview toggle is unavailable in this
workspace, fall back to the bootcamp's taught CDC method — architecture doc §20 condition 1).

Decisions taken where the above is silent (C-2 implementation):
- REPLICA IDENTITY FULL on both CDF tables, in the DDL. Postgres logical decoding emits only the
  primary key for an UPDATE or DELETE under the default replica identity, so a removed ticker
  would arrive in the Delta history table with no ticker and no added_by — the CDC demo step
  would show a row of nulls. It requires table ownership; that is a grant to fix, not a statement
  to drop.
- Ids are application-generated TEXT, not serials or gen_random_uuid() defaults, because
  save_research_report has to return the id it wrote (C-4) and generating it in Python keeps that
  a plain INSERT. The demo user and watchlist get readable, stable ids as a side effect.
- ensure_tables() EXECUTES setup/create_lakebase.sql rather than holding a second copy of the
  statements — the same single-source rule that emptied sql/ (B0).
- Column types: TEXT and TIMESTAMPTZ throughout, real foreign keys inside Lakebase, and NO
  foreign key on research_reports.forecast_id, which points at a Delta row in the other store.
- The schema name is a constant in code, not an environment variable: it is an identifier, so it
  cannot be a bound parameter, and an env-driven identifier spliced into SQL is exactly what
  "parameterized SQL only" exists to prevent. app.yaml's LAKEBASE_SCHEMA is informational.
- Every function takes an optional connection. Passing one skips the pool, which is how the unit
  tests assert the SQL and its parameters without a database; the pool's context manager owns the
  transaction otherwise.

### C-3 llm/call_model.py + telemetry.py

    def call_model(task: str, messages, tools=None, response_format=None) -> Response

Reads endpoint name from config by task ("agent" now; "slm" later). Wraps the
Databricks Foundation Model API (OpenAI-compatible chat completions). telemetry.py
appends {ts, task, model, latency_ms, ok, in_tokens, out_tokens} to a Delta table
gold.model_calls. No routing logic. No tiers.

Decisions taken where the above is silent (C-3 implementation):
- Transport is a plain POST to {host}/serving-endpoints/{endpoint}/invocations with requests,
  which is already a dependency. databricks-sdk's serving_endpoints.query takes typed ChatMessage
  objects and would mean translating OpenAI-shaped messages/tools/response_format into SDK
  dataclasses and back; the SDK is still what resolves the CREDENTIAL, so notebooks, jobs and the
  app all authenticate as the identity they already run as.
- Retries: 429 and 5xx only, 3 attempts, exponential backoff with jitter — a scale-to-zero
  endpoint answers 503 while it wakes. Fewer attempts than the ingestion client's five because a
  model call sits on the interactive path.
- Nothing credential-bearing and no request or response body reaches a log record: prompts and
  completions carry user text. The endpoint, status, latency and token counts do.
- gold.model_calls carries a call_id (uuid4) beyond the spec's column list. Rule 4 requires a
  MERGE on declared keys and the record has no natural identity, so without it a retried flush
  would duplicate rows. Failed calls are rows too: ok=false with NULL tokens.
- record() only buffers; flush() writes. A Spark write must not sit inside a chat turn, and a
  failed flush re-queues its records — safe precisely because the write is a MERGE on call_id.
- Three modes (telemetry.mode, above), because the Streamlit app cannot run Spark. The unset
  default is log-only: a default that needs a SparkSession turns a forgotten setting into an
  agent failure.

### C-4 agent/
tools.py — four functions with JSON-schema declarations:
  get_market_forecast(ticker) → latest gold.forecast_runs + regime_states row
  search_market_news(ticker, query, k=5) → AI Search hybrid results
  update_watchlist(action: add|remove, ticker) → lakebase write, returns new list
  save_research_report(ticker, question, report_md) → lakebase write, returns id
agent.py — plain tool-calling loop (max ~6 tool iterations): send system prompt +
history + tool schemas via call_model("agent", ...); execute returned tool calls;
append results; stop when the model returns a final text answer.
prompts.py — system prompt requirements: role (market research explainer); MUST call
get_market_forecast before making quantitative claims; MUST ground news claims in
search results and mention article titles; NEVER invent numbers; NEVER give buy/sell
advice; confirm writes it performs; state the news-decay assumption when explaining a
forecast's news conditioning.

Decisions taken where the above is silent (C-4 implementation):
- Gold is read over the SQL WAREHOUSE, not Spark: the app hosting the agent has no SparkSession.
  src/database/delta.py implements that path (databricks-sql-connector, :name parameters bound
  server-side, one connection per call). Identifiers come from config; every value is bound.
- get_market_forecast takes the most recent as_of_date and, when several models wrote that date,
  prefers news_markov > markov > gbm. The spec says any model_used; ordering by the column
  alphabetically would systematically surface gbm, the bottom rung of the fallback ladder. The
  regime row is matched on the forecast's OWN (ticker, as_of_date), so the pair always describes
  the same fit.
- A ticker with no forecast returns {"found": false, ...} rather than raising. "No forecast has
  been computed for AMD yet" is a true answer the agent must be able to give, and it is the
  normal state for a ticker the CDC demo added minutes ago. Same for a search with no hits.
- Every tool returns a JSON-serializable dict: the result becomes a tool message, and a date or a
  Decimal would fail at serialization time, mid-turn.
- Tickers are validated (the same rule as lakebase.py) before any query or write, so "add tesla
  to my list" fails as an argument error the model can correct rather than as a stored row.
- The loop treats a TOOL failure as a result — the error text goes back to the model, which can
  retry or explain — and lets a MODEL failure propagate, since there is nothing to recover with.
- The iteration cap ends the turn with an explicit "I could not finish" and sets
  hit_iteration_limit. The failure it prevents is a confident answer assembled from nothing.
- Telemetry is not recorded by the loop: call_model already writes one record per call (C-3), and
  a second record per turn would double-count. test_agent_loop.py asserts the records exist by
  running the real call_model over a fake HTTP session.
- The system prompt interpolates news.half_life_days from config rather than stating "2 days" in
  prose, so changing the config cannot leave the agent describing the old behaviour.

### C-5 Streamlit app
Reads Delta via databricks-sql-connector (serverless SQL warehouse), Lakebase via the
psycopg v3 pool from C-2 (db.py wrapped by src/database/lakebase.py). Page data sources:
  market_research  → daily_prices (chart), regime_states, forecast_runs, news_articles
  research_agent   → agent loop; watchlist sidebar from Lakebase
  model_evaluation → backtest pooled summary; ALWAYS renders n and fallback rate;
                     verdict line computed from the numbers (three cases: better /
                     worse / indistinguishable at this n)
app.yaml declares env (warehouse id, catalog, Lakebase conn). Cache reads with
st.cache_data(ttl=600).

### C-6 Workflow
setup/create_workflow.py: 7 tasks in the A1 order, daily schedule, retries=2 on
ingestion tasks, sync_news_index last.

### C-7 Checkpoint C tests
test_agent_tools — each write tool call produces the expected Lakebase row
(integration test against real Lakebase); read tools return schema-valid payloads.
Manual end-to-end: the A4 demo script, once, before calling C frozen.

The live round trip is kept runnable and opt-in (AGENT_LIVE_TEST=1), like the Lakebase one, since
it needs a warehouse, an index and a Postgres role. The fake-backed tests alongside it assert
what an integration test cannot: the exact SQL text and its bound parameters, the index filter
document, and the four JSON schemas against their handlers' signatures — a drifted schema fails
at model-call time, and a value interpolated into a query still returns the right row.

## CHECKPOINT D — Polish only
README with architecture diagram; screenshots; demo script; telemetry mini-view;
optional SLM classifier (call_model("slm"), JSON-schema-constrained event_type enum,
written to a new column/table, surfaced as UI labels + AI Search filter, NEVER into
Model C).

---

# PART C — DATABRICKS DEPLOYMENT WORKFLOW (the part Cursor can't infer)

## C-a Dev loop: Cursor ↔ GitHub ↔ Databricks Git folder
1. Create the GitHub repo; connect Databricks Git folder (Repos) to it.
2. Cursor edits locally → commit/push → in Databricks, pull the Git folder.
3. Notebooks in notebooks/ are thin: %pip install -r requirements, sys.path append
   the repo root, import from src/, call functions. ALL logic lives in src/*.py so
   Cursor owns it and tests run locally with pytest (pure-pandas parts) without a
   cluster.
4. Iterate: logic bugs → fix locally, push, pull, re-run notebook. Spark-specific
   issues → debug in the notebook, then port the fix back into src/.
5. Local secrets: copy .env.example to .env and fill in real values. .env is git-ignored and
   local-only — Databricks jobs read their secrets from secret scopes (A-0) and the deployed
   app gets configuration from app resources / app.yaml, so neither reads .env. python-dotenv
   is in requirements.txt for this and is deliberately absent from requirements-databricks.txt
   and app/requirements.txt.

## C-b Where Spark actually runs
Only ingestion/pipelines use Spark. Modeling code must not import pyspark. The
boundary is exactly one line: pdf = spark.table(f"{cat}.silver.daily_features")
.where(col("ticker")==t).orderBy("trade_date").toPandas(). Data volume (≈2.5k rows
per ticker) makes this trivially safe. Do not partition these tiny Delta tables;
defaults are fine.

## C-c Environment installs
Serverless/interactive notebooks: %pip install statsmodels exchange_calendars at the
top, or set the job environment's dependencies so workflow tasks get them. The
Streamlit app gets its packages from the app's requirements.txt automatically.

## C-d Verify-first list (run these checks before building on each)
□ Massive smoke test 200 (A-0)                  □ secret scope readable in notebook
□ SQL warehouse exists (app needs it)           □ AI Search endpoint quota available
□ Lakebase project creatable                    □ Lakebase CDF toggle present
□ Foundation Model endpoint responds to one call □ Git folder pulls from repo

## C-e Known failure modes → fastest diagnosis
429s during backfill → your throttle isn't actually sleeping; log inter-request gaps.
MarkovRegression all-NaN params → returns not scaled to %; or NaNs left in endog.
"lengths differ" on exog_tvtp → the shift(1) row wasn't dropped jointly with endog.
Backtest wildly good → you leaked: check smoothed-probs grep test and TVTP lag test.
App can't read Delta → warehouse id/permissions in app.yaml, not your code.
Index empty after sync → CDF property missing on news_articles at creation time.
Notebook kernel dies, exit 134, no traceback → something imported psycopg on serverless. It
  SIGABRTs in libpq at import; Lakebase is app-container-only (see the B0 environment note).
Ticker-filtered news search finds nothing for a multi-ticker article → doc_id is NULL or the
  index is keyed on article_id; re-run build_silver_news, then trigger a sync.

END OF SPEC. Build order within each checkpoint follows document order.