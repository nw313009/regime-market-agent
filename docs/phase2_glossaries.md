# Phase 2 — Knowledge Base
## Regime-Aware Market Intelligence Agent (frozen v2.1)

Scope rule: every term below is load-bearing in YOUR system. Each entry gives the plain meaning, the math where one exists, and where it appears in your build. If a term isn't here, you don't need it to finish or defend this capstone.

---

# PART 1 — TECHNICAL GLOSSARY

## 1.1 Databricks platform

**Delta Lake / Delta table** — A storage format: Parquet data files plus a transaction log. The log is what upgrades plain files into a database-like table with ACID guarantees, time travel, and schema enforcement.
*In your system:* every Bronze/Silver/Gold table is a Delta table.

**ACID transactions** — Atomicity (a write fully happens or doesn't at all), Consistency (table never left half-valid), Isolation (concurrent readers don't see partial writes), Durability (committed = permanent). Delta gives you this on data-lake files.
*In your system:* why your Streamlit app can read `forecast_runs` while the daily job is writing it, without seeing garbage.

**Unity Catalog** — Databricks' governance layer: a three-level namespace `catalog.schema.table` (e.g., `market_intel.silver.daily_prices`) plus permissions, lineage, and ownership on top of it.
*In your system:* every table name you write. `market_intel` is the catalog, `bronze/silver/gold` are schemas.

**Medallion architecture (Bronze/Silver/Gold)** — A convention, not a technology. Bronze = raw as-ingested (replayable, ugly). Silver = cleaned, deduplicated, typed, conformed. Gold = business-ready analytical products.
*In your system:* Bronze `prices_raw`/`news_raw` → Silver `daily_prices`/`news_articles`/`daily_features` → Gold `forecast_runs`/`regime_states`/`backtest_metrics`. The interview answer for "why three layers": you can rebuild Silver from Bronze when logic changes, without re-calling the API.

**Delta Change Data Feed (Delta CDF)** — A table property (`delta.enableChangeDataFeed = true`) that makes Delta record row-level changes (insert/update/delete + version) so downstream consumers can read *only what changed* instead of rescanning the table.
*In your system:* required on `silver.news_articles` because the AI Search Delta Sync index uses CDF to update itself incrementally.

**Lakebase** — Databricks-managed Postgres for transactional/operational workloads: low-latency single-row reads/writes, the thing Delta is bad at.
*In your system:* `users`, `watchlists`, `watchlist_tickers`, `research_reports`. Authoritative for operational state only.

**WAL (Write-Ahead Log)** — Postgres durability mechanism: every change is appended to a sequential log *before* being applied to tables. Because the WAL is a complete ordered record of all changes, it's also the natural tap point for replication and CDC.
*In your system:* Lakebase CDF reads the WAL to capture your watchlist writes.

**CDC (Change Data Capture)** — The general pattern of capturing changes from an operational database and delivering them to an analytical system, rather than re-copying whole tables.
*In your system:* Lakebase CDF is your CDC implementation: agent writes watchlist → Postgres WAL → Lakebase CDF → Delta history table. This is bootcamp requirement #6 and your best demo moment.

**Lakebase CDF vs Delta CDF** — Same acronym, two mechanisms, opposite directions. Lakebase CDF: Postgres → Delta (your operational CDC). Delta CDF: change tracking *within* Delta tables (feeds your AI Search index sync). Never conflate them in the demo.

**Databricks Workflow (Job)** — The orchestrator: a DAG of tasks with a schedule, retries, and dependencies.
*In your system:* the daily pipeline — ingest_prices → ingest_news → build_silver → build_features → fit_models → run_forecasts → sync_news_index — scheduled once daily with a manual "Run now" for the demo.

**Databricks App** — Databricks-hosted web application (your Streamlit code) running inside the workspace with governed access to Unity Catalog and Lakebase.
*In your system:* the three-page frontend.

## 1.2 Spark

**DataFrame** — Spark's distributed table abstraction: rows + named typed columns, partitioned across executors. You express transformations on it; Spark plans the execution.

**Transformation vs action** — Transformations (`select`, `filter`, `withColumn`, `groupBy`) *describe* work and return a new DataFrame instantly. Actions (`write`, `count`, `collect`) *trigger* work.

**Lazy evaluation** — Spark accumulates your transformations into a plan and executes nothing until an action forces it, so it can optimize the whole plan at once. Practical consequence: an error in a transformation may only surface at the `write` line.
*In your system:* your feature pipeline is a chain of transformations executed by the final write to `daily_features`.

**Window function** — A computation over a sliding set of rows relative to each row — exactly what rolling features are.
*In your system:* `realized_vol_20d` and `volume_zscore_20d` are window functions partitioned by ticker, ordered by trade_date, over the trailing 20 rows. This is the core Spark skill your project demonstrates.

**Schema enforcement** — Delta rejects writes whose columns/types don't match the table. Your friend: a malformed API response fails loudly at Bronze instead of corrupting Silver silently.

**MERGE (upsert)** — "Update if the key exists, insert if it doesn't" in one atomic statement.
*In your system:* how re-running ingestion for a date range doesn't create duplicate rows — MERGE on `(ticker, trade_date)` for prices and `(article_id, ticker)` for news. This is what makes the pipeline *idempotent*.

**Idempotency** — Running an operation twice leaves the same state as running it once. The single most important property of your pipeline: it means a failed job can simply be re-run.

## 1.3 AI / retrieval

**Token** — The unit models read and bill in; roughly ¾ of an English word. Your telemetry logs input/output token counts per call.

**Embedding** — A learned function mapping text → a vector (list of ~1,000 floats) such that semantically similar texts map to nearby vectors. "Chip demand surges" and "semiconductor sales jump" share almost no words but land close together.
*In your system:* AI Search computes embeddings of `embedding_text` (title + description) automatically — managed embeddings, you never call an embedding model yourself.

**Cosine similarity** — The standard nearness measure for embeddings: cos(θ) = (a·b)/(‖a‖‖b‖), ranging 1 (same direction) to −1 (opposite). Retrieval = "return the k vectors with highest cosine similarity to the query vector."

**Vector index** — A data structure making that nearest-neighbor search fast without comparing against every stored vector.

**Keyword (lexical) search / BM25** — Classic exact-term matching with frequency-based scoring. Finds "NVDA" and "Q3 guidance" literally; misses paraphrases.

**Hybrid retrieval** — Runs semantic (vector) and keyword search together and merges rankings. You get paraphrase recall AND exact-ticker precision.
*In your system:* your index's search mode. Interview answer for "why hybrid": financial queries mix semantic intent ("why is volatility elevated") with exact identifiers ("NVDA").

**Delta Sync index** — An AI Search index bound to a Delta source table that updates itself incrementally from the table's CDF instead of being rebuilt.
*In your system:* one index on `silver.news_articles`, TRIGGERED sync as the last workflow task.

**RAG (Retrieval-Augmented Generation)** — Pattern: retrieve relevant documents first, put them in the model's context, have it answer *grounded in* them, reducing hallucination.
*In your system:* the agent's `search_market_news` tool is the R; its explanation is the AG.

**Foundation Model API** — Databricks-hosted model endpoints, pay-per-token, OpenAI-compatible interface. Why you deploy no model servers.
*In your system:* everything behind `call_model(task, ...)`.

**Function calling (tool use)** — You send the model a JSON description of available tools; instead of prose it can return a structured request like `{"tool": "get_market_forecast", "arguments": {"ticker": "NVDA"}}`. Your code executes it, returns the result, and the model continues. The loop — model proposes → code executes → model observes → repeat — IS your agent.
*In your system:* the four tools. Architectural point to say in interviews: the model never computes anything; it only chooses which deterministic function runs.

**Structured outputs / JSON schema** — Constraining a model's response to match a declared schema, so output is machine-parseable by construction.
*In your system:* the stretch-goal SLM classifier (event_type must be one of your 8 enum values).

**System prompt** — Standing instructions defining the agent's role, tool policy, and boundaries ("you explain forecasts, you never invent numbers, you cite retrieved articles"), separate from the user's question.

**SLM (small language model)** — A small, cheap, fast model for constrained repetitive tasks. Stretch goal only; UI labels and search filters, never Model C.

## 1.4 Engineering practice

**REST API / endpoint / HTTP status codes** — REST: request-response over HTTPS against resource URLs (endpoints). Codes you'll actually handle: 200 success, 401 bad/missing API key, 403 plan doesn't include this data, 404 bad path, 429 rate-limited, 5xx their problem (retry).
*In your system:* `massive_client.py` handles all of these explicitly.

**Pagination** — APIs return large result sets in pages with a cursor/next-URL. Your news backfill must follow next-links until exhausted, not just take page one.

**Throttling / retry with exponential backoff** — Throttling: your client deliberately spacing requests to stay under the provider's rate limit. Backoff: on failure, wait then retry with growing delays (1s, 2s, 4s...), plus jitter (small randomization) so retries don't synchronize.

**Secret management** — API keys live in Databricks Secrets, referenced by scope/key at runtime. Never in code, never in Git, never in notebook output.
*In your system:* the Massive key. An interviewer seeing a key in your repo ends the interview.

**Config-driven** — Behavior (model name, tickers, paths, rate limits, seed) read from `config.yaml`, not hard-coded. Your frozen doc mandates this for model access and rate assumptions.

**Unit / integration / smoke test** — Unit: one function's logic in isolation (columns of P sum to 1). Integration: components together against real infrastructure (agent write actually lands in Lakebase). Smoke: cheapest possible "is it alive" check (Massive returns 200). Your Monday morning starts with the smoke test.

**Regression test** — A test that pins a specific bug fixed forever. Your no-lookahead test is one: mutate news after T, assert the forecast at T is unchanged.

**Telemetry** — Per-call operational measurements (timestamp, task, model, latency, tokens, success) appended to a log/table. Instrumentation, not a dashboard project.

**Primary / composite key** — Column(s) uniquely identifying a row; composite = several columns jointly. `(article_id, ticker)` on news because one article ↔ many tickers.

---

# PART 2 — FUNCTIONAL GLOSSARY

## 2.1 Market data

**Ticker** — Exchange symbol for a security (NVDA). Your universe: 5 seeded + user watchlist entries.

**OHLCV** — One trading session summarized as Open, High, Low, Close prices + Volume (shares traded). One row per ticker per day; Massive calls these aggregate bars.
*In your system:* the schema of `silver.daily_prices`.

**VWAP** — Volume-Weighted Average Price: Σ(pᵢ·vᵢ)/Σvᵢ over the session — the "typical" price weighted by how much actually traded there. Massive supplies it; you store it; no model uses it in v1.

**Trading day / market calendar** — Equity markets close weekends and exchange holidays. A market calendar is the authoritative list of open sessions. Frozen doc: weekend/holiday news maps to the *next session via the calendar*, never via naive weekday math (Good Friday breaks weekday math).

## 2.2 Returns and volatility

**Simple vs log return** — Simple: Rₜ = (Sₜ − Sₜ₋₁)/Sₜ₋₁. Log: rₜ = ln(Sₜ/Sₜ₋₁). Two reasons models prefer log: they **add across time** (r over 5 days = Σ daily r, while simple returns compound multiplicatively), and if prices follow GBM, log returns are exactly the normally distributed object.
*In your system:* everything downstream of `daily_features.log_return` — both Markov models take log returns as endog, and MC accumulates them by summation then exponentiates once.

**Realized volatility** — Sample standard deviation of recent log returns:
σ̂ = sqrt( (1/(n−1)) Σ (rᵢ − r̄)² ) over a trailing window (yours: 20 days). "Realized" = measured from what happened, vs "implied" (option-market-derived — not in your project). Annualized by ×√252 if you ever display it that way (252 ≈ trading days/year).
*In your system:* `realized_vol_20d` — a UI/context feature; the regimes themselves estimate their own σ via MLE, they don't consume this column.

**Momentum** — Recent trend measure, e.g. 5-day cumulative return. Feature-table resident, model-absent in v1. Do not sneak it into the models.

**Z-score** — Standardization: z = (x − μ)/σ over a window. `volume_zscore_20d` answers "is today's volume unusual relative to this ticker's own recent normal" — comparable across tickers with wildly different absolute volume.

**Sentiment score** — Massive's vendor-supplied per-article label, normalized by you to +1/0/−1, averaged per ticker-day, then decayed:
N_t = (1.0·s_t + 0.5·s_{t−1} + 0.25·s_{t−2}) / 1.75
No-news days: s = 0 (neutral), with `news_count` preserved so the UI can distinguish "neutral news" from "no news."
*In your system:* `news_sentiment_3d` — the ONLY covariate entering Model C.

## 2.3 The stochastic models

**Stochastic process** — A quantity evolving randomly over time; a model of it specifies the distribution of each step, not the value.

**Brownian motion (Wiener process)** — The canonical continuous random walk W_t: independent normal increments, W_{t+Δ} − W_t ~ N(0, Δ). The randomness source inside GBM.

**GBM (Geometric Brownian Motion)** — Price model dS = μS dt + σS dW: percentage changes (not absolute changes) have constant drift μ and diffusion σ. Its discrete consequence, which is all you implement:
r_{t+1} ~ N(μ, σ²),  S_{t+1} = S_t · e^{r_{t+1}}
with one (μ, σ) estimated from the training window.
*In your system:* Model A. Its whole job is to be the honest baseline the regime models must beat.

**Drift / diffusion** — μ = deterministic tendency per unit time; σ = magnitude of randomness per unit time. In regime models each regime has its own pair: (μ₀, σ₀), (μ₁, σ₁).

**Markov chain / Markov property** — A state process where the next state's distribution depends only on the current state: P(Z_{t+1} | Z_t, Z_{t−1}, ...) = P(Z_{t+1} | Z_t). The "memorylessness" that makes the model tractable.

**Regime (hidden/latent state)** — Z_t ∈ {0,1}: which volatility mode the market is in. *Latent* = never directly observed; inferred probabilistically from returns. Labels assigned AFTER fitting by sorting on fitted variance (label switching defense — see 2.5).

**Transition probability / matrix** — p_{ij} = P(next regime | current regime), collected into P. Frozen-doc orientation rule: statsmodels builds P with rows = next state, columns = previous state, **columns sum to 1** (left-stochastic). Your sampler and tests are written against that orientation.

**Stationary distribution** — π with (row convention) π = πP: the long-run share of time in each regime. Your orientation unit test simulates long chains and checks empirical state frequencies against π — a transposed matrix fails this immediately.

**TVTP (time-varying transition probabilities)** — Transitions as a function of an exogenous signal instead of constants:
P(Z_{t+1} = j | Z_t = i, N_t)
implemented as a logistic function of N: p = e^{β₀+β₁N}/(1+e^{β₀+β₁N}), which maps any real number into (0,1). statsmodels' `exog_tvtp` does this; you must pass `[ones, N.shift(1)]` — the lag is the frozen alignment rule.
*In your system:* the entire difference between Models B and C, and the heart of your research question.

**Exogenous variable** — Input from outside the modeled system (news N_t), as opposed to endogenous (the returns being modeled).

**News decay assumption** — Future news is unknown at forecast time, so current news is decayed rather than invented: N_{t+h} = N_t·e^{−λh} with half-life 2 trading days ⇒ λ = ln2/2. An explicitly disclosed modeling assumption, stated in the UI.

## 2.4 Estimation and inference

**Likelihood / MLE** — Likelihood L(θ) = probability of the observed returns given parameters θ. Maximum Likelihood Estimation picks θ maximizing it. How every model here is fitted.

**EM algorithm** — The iterative MLE workhorse for latent-variable models: E-step (infer regime probabilities given current θ), M-step (re-estimate θ given those probabilities), repeat until the likelihood stops improving.

**Convergence / non-convergence / local maxima / multiple starts** — Converged = iterations settled. Non-converged = hit the iteration cap while still moving ⇒ parameters untrustworthy ⇒ fallback ladder fires (C→B→A, model recorded per forecast). The likelihood surface has multiple hills; starting from several random points and keeping the best fit guards against settling on a minor one.

**Degenerate fit** — Converged to a useless answer: σ₀ ≈ σ₁ (regimes indistinguishable) or a transition probability ≈ 1 (an absorbing state the chain never leaves). Detect and treat as failure for ladder purposes.

**Label switching** — Across refits, which regime gets index 0 is arbitrary; parameters are fine, names flip. Defense: after every fit, sort regimes by fitted variance and reorder ALL dependent quantities (means, sigmas, transition matrix rows AND columns, probability vectors) consistently.

**Hamilton filter** — The forward recursion computing, day by day, P(Z_t | r_1..r_t): propagate yesterday's regime probabilities through P, then Bayes-update on today's return. Runs inside `.fit()` and produces the filtered probabilities.

**Filtered vs smoothed probabilities** — Filtered: P(Z_t | data ≤ t) — uses only the past; produced by the Hamilton filter. Smoothed: P(Z_t | ALL data) — a backward pass (Kim smoother) that conditions on the future too. Frozen rule: filtered everywhere in forecasting and backtesting; smoothed permitted only for retrospective history charts. At the final date T of a sample they coincide — which is why live forecasting is safe either way but backtests are not.

## 2.5 Simulation and evaluation

**Monte Carlo simulation** — Answering distribution questions by sampling: simulate the 5-day future 5,000 times, read the distribution off the outcomes. Per path per day: decay news → build transition matrix from fitted TVTP params → sample next regime → draw return ~ N(μ_regime, σ²_regime) → compound price. Seeded (config) for reproducibility.

**Percentile / P10, P50, P90** — Value below which X% of simulated outcomes fall. P50 = median. [P10, P90] = your 80% prediction interval.

**Prediction interval vs confidence interval** — Prediction interval: range for a future *observation* (your [P10,P90] for the 5-day return). Confidence interval: range for an estimated *parameter*. Interviewers conflate these; you shouldn't.

**Calibration** — Do stated probabilities match observed frequencies? Among all days you said P(positive) ≈ 60%, did ~60% actually close positive? The property your evaluation page actually measures.

**Brier score** — Mean squared error of probability forecasts against binary outcomes:
BS = (1/n) Σ (p_i − o_i)², o_i ∈ {0,1}
for your P(5-day return > 0). Lower is better; 0.25 = the score of always saying 50%.

**MAE (median-return error)** — (1/n) Σ |r̂_median,i − r_actual,i|. The interviewer-friendly metric.

**Interval coverage** — Fraction of actual outcomes landing inside the stated 80% interval. Well-calibrated ⇒ ≈ 80%. Below ⇒ overconfident (intervals too narrow); above ⇒ underconfident.

**Walk-forward backtest / origin / temporal isolation** — Stand at historical date T (an *origin*), fit on data ≤ T only, forecast T+5, score against reality, advance T, repeat. Temporal isolation: nothing after T touches fitting, features, regime probabilities, or TVTP inputs. Your origins: weekly, ~26 weeks, ≥252-day training window.

**Lookahead bias** — Any future information leaking into a "past" decision; inflates backtest results and is the most common fraud-adjacent error in quant work. Your three specific defenses: lagged TVTP input, filtered-only probabilities, the mutate-future-news regression test.

**Pooling / sample size / standard error** — 26 origins/ticker is tiny; pool across 5 tickers ⇒ n = 130 headline metrics, with n displayed. SE of a proportion p̂: sqrt(p̂(1−p̂)/n) — at p̂=0.8, n=26 that's ±7.8 points, which is why per-ticker coverage differences are noise. The evaluation page must be able to truthfully print "no meaningful improvement detected at this sample size."

---

## Ownership check — how you'll know Part 1 and 2 stuck

You own this material when you can answer, cold, without the doc:

1. Why log returns instead of simple returns, both reasons?
2. Trace one watchlist write from agent tool call to Delta history, naming every mechanism it passes through.
3. Why must `exog_tvtp` carry lagged news, and what breaks if it doesn't?
4. Why is a smoothed-probability backtest lying to you, and why is the live forecast exempt?
5. What does "columns sum to 1" imply about how you index the matrix when sampling the next regime?
6. Why can't your evaluation table honestly claim a 0.014 Brier improvement is meaningful?

These six ARE interview questions. Not rhetorical ones.