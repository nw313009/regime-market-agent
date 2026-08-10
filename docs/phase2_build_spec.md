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
├── sql/            (DDL kept in sync with setup/)
├── tests/          test_features.py | test_models.py | test_monte_carlo.py |
│                   test_no_lookahead.py | test_agent_tools.py | test_idempotency.py
├── config/config.yaml
├── requirements.txt              # local 3.12 venv, fully pinned (dev + pytest)
└── requirements-databricks.txt   # ONLY packages the Databricks runtime lacks:
                                  # statsmodels, exchange_calendars,
                                  # databricks-sql-connector, psycopg2-binary

Dependency environments are split three ways, recorded here under rule 2 (spec silent →
simplest working option). requirements.txt fully pins the local Python 3.12 dev/test venv —
statsmodels, pandas, numpy, exchange_calendars, databricks-sql-connector, psycopg2-binary,
streamlit, requests, pyyaml, pytest — with statsmodels==0.14.6 as a hard floor, since earlier
versions fail to import under pandas 3.0; it is never installed on a cluster.
requirements-databricks.txt is what notebooks and job environments install, and lists only the
four packages the runtime lacks (statsmodels, exchange_calendars, databricks-sql-connector,
psycopg2-binary): NEVER pip-upgrade pandas, numpy or pyarrow on a cluster, because the runtime
pins those three against its own Spark build and replacing them breaks Spark in ways that
surface far from the change. app/requirements.txt carries what the Databricks App consumes
(streamlit, databricks-sql-connector, psycopg2-binary, pyyaml, requests); the app has no
SparkSession and fits no models, so it needs no statsmodels. The frozen architecture doc is
silent on dependency files and is unaffected by this split.

config.yaml keys: massive.base_url, massive.rate_limit_per_min, tickers.seed[5],
forecast.horizon_days=5, forecast.n_paths=5000, forecast.seed=42,
news.half_life_days=2, backtest.min_train_days=252, backtest.origin_freq=weekly,
backtest.n_weeks=26, model.agent_endpoint, model.slm_endpoint (unused until stretch),
catalog=market_intel.

## CHECKPOINT A — Data works

### A-0 Smoke test (FIRST ACTION, before anything else)
Notebook cell, near-verbatim:

    import requests
    key = dbutils.secrets.get(scope="capstone", key="massive_api_key")
    r = requests.get(f"{BASE_URL}/v2/aggs/ticker/NVDA/range/1/day/2026-07-01/2026-08-01",
                     params={"apiKey": key}, timeout=30)
    print(r.status_code); print(r.json() if r.ok else r.text[:500])

(Adjust path to Massive's current aggregates route — check their docs, don't trust
memory.) 200 + JSON → proceed. Anything else → stop and diagnose (401/403 = key/plan,
connection error = workspace egress). Secret setup first:
databricks secrets create-scope capstone; databricks secrets put-secret capstone
massive_api_key.

### A-1 massive_client.py contract

    class MassiveClient:
        def __init__(cfg, secret_getter): ...
        def get_daily_aggregates(ticker, start_date, end_date) -> list[dict]
        def get_news(ticker, published_after) -> list[dict]

Requirements: token-bucket or simple sleep-based throttle from
cfg.rate_limit_per_min; follow pagination cursors/next_url to exhaustion; retry on
429/5xx with exponential backoff + jitter (max 5 attempts); raise on 401/403 with a
clear message; log every request (url minus key, status, latency).

### A-2 Bronze
bronze.prices_raw, bronze.news_raw: store the近-raw payload rows plus
source, ingested_at, request_id, ticker, source_timestamp. bronze.ingestion_runs:
run_id, task, started_at, finished_at, status, rows_written, error. MERGE keys:
prices (ticker, source_timestamp); news (article_id, ticker) after a light explode.

### A-3 Silver
silver.daily_prices — ticker, trade_date, open, high, low, close, volume, vwap.
MERGE on (ticker, trade_date). Convert Massive's epoch-ms timestamps to the trading
date in exchange timezone (America/New_York), not UTC-naive.

silver.news_articles — article_id, ticker, published_at, title, description,
publisher, sentiment_label, sentiment_score, embedding_text (= title + "\n" +
description), article_url. Composite key (article_id, ticker): explode the tickers
array — one row per (article, ticker). MERGE on the composite key.
Enable CDF at creation:
  TBLPROPERTIES (delta.enableChangeDataFeed = true)

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
  s_t               = mean of per-article normalized sentiment that session
                      (positive→+1, neutral→0, negative→−1); 0 if no articles
  news_sentiment_3d = (1.0·s_t + 0.5·s_{t−1} + 0.25·s_{t−2}) / 1.75
  news_count        = article count mapped to that session
Spark window functions partitioned by ticker ordered by trade_date. Rows with null
rolling features (warm-up period) stay in the table; the modeling layer drops them.

### A-5 Checkpoint A tests
test_features: known synthetic price series → exact expected log_return, vol, z.
test_weekend_news: Saturday+Sunday articles land in Monday's s_t and news_count.
test_idempotency: run silver build twice on same Bronze → identical row counts and
checksums.

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
test_mc_seed — same seed → identical percentiles; different seed → different.
test_fallback — inject a FitError from C; assert B used and recorded.

## CHECKPOINT C — Product works

### C-1 AI Search
setup/create_ai_search.py: create endpoint (one, STANDARD); create Delta Sync index
market_intel.silver.news_index on silver.news_articles, embedding source column
embedding_text (managed embeddings), sync mode TRIGGERED. sync_news_index task in the
workflow triggers it. Query path: hybrid search, filter by ticker, top_k≈5.

### C-2 Lakebase
setup/create_lakebase.sql: users(user_id, display_name, created_at);
watchlists(watchlist_id, user_id, name, created_at);
watchlist_tickers(watchlist_id, ticker, added_at, added_by, PRIMARY KEY
(watchlist_id, ticker)); research_reports(report_id, user_id, ticker, question,
report_md, forecast_id, created_at).
database/lakebase.py: connection via psycopg2 using workspace-provided credentials
(env vars in app.yaml); small functions: add_ticker, remove_ticker, save_report,
get_watchlist. Parameterized SQL only.
Enable Lakebase CDF on watchlist_tickers and research_reports → Delta history
tables (per current docs; if the preview toggle is unavailable in this workspace,
fall back to the bootcamp's taught CDC method — architecture doc §20 condition 1).

### C-3 llm/call_model.py + telemetry.py

    def call_model(task: str, messages, tools=None, response_format=None) -> Response

Reads endpoint name from config by task ("agent" now; "slm" later). Wraps the
Databricks Foundation Model API (OpenAI-compatible chat completions). telemetry.py
appends {ts, task, model, latency_ms, ok, in_tokens, out_tokens} to a Delta table
gold.model_calls. No routing logic. No tiers.

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
advice; confirm writes it performs.

### C-5 Streamlit app
Reads Delta via databricks-sql-connector (serverless SQL warehouse), Lakebase via
psycopg2. Page data sources:
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

END OF SPEC. Build order within each checkpoint follows document order.