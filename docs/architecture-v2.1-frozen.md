# Regime-Aware Market Intelligence Agent

## Frozen Architecture — v2.1-final

**Status: FROZEN**

Architecture changes are prohibited unless explicitly authorized by the project owner.

The project is intentionally optimized for **technical depth, end-to-end completion, and interview defensibility**, not maximum platform surface area.

---

## 1. System Architecture

```text
Massive API
   │
   ├── Prices
   └── Financial News
          │
          ▼
      Bronze Delta
          │
          ▼
      Silver Delta
   prices / news /
   daily_features
          │
          ▼
    PySpark Pipeline
          │
    ┌─────┼──────────────┐
    │     │              │
    ▼     ▼              ▼
   GBM   Markov      News-Markov
    A       B             C
    │       │             │
    └───────┼─────────────┘
            ▼
       Monte Carlo
   5,000 paths × 5 days
            │
            ▼
        Gold Delta
 forecasts / regimes /
  backtest metrics
       │          │
       │          └──────────────► Streamlit Databricks App
       │                                 │
       ▼                                 │
   AI Search                             │
       │                                 │
       └────────► Research Agent ◄───────┘
                       │
                   4 tools
                       │
                       ▼
                   Lakebase
          users / watchlists / reports
                       │
                 Lakebase CDF
                       │
                       ▼
                  Delta history
```

---

# 2. Data Responsibilities

## Delta / Unity Catalog

Authoritative for analytical data:

* raw price data
* raw financial news
* cleaned price data
* normalized news
* engineered daily features
* fitted regime outputs
* forecast outputs
* historical forecasts
* backtest results
* model-comparison metrics

## Lakebase

Authoritative only for application/transactional state:

* users
* watchlists
* watchlist membership
* saved research reports

Operational changes flow:

```text
Agent / App
    ↓
Lakebase
    ↓
Lakebase CDF
    ↓
Delta history
```

There is **no Delta → Lakebase forecast-serving synchronization in this capstone**.

The Databricks App reads the small Gold analytical result set directly.

---

# 3. Data Pipeline

Primary ingestion source:

**Massive REST API**

Massive supports historical OHLC aggregation through its Stocks REST API and exposes financial-news records containing publication timestamps, article IDs, associated tickers and sentiment insights.

Pipeline:

```text
Massive
   ↓
Bronze
   ↓
Silver
   ↓
Spark feature engineering
   ↓
Gold
```

The Massive client must include:

* pagination
* retries
* retry/backoff handling
* deliberate request throttling
* request logging
* secure API-key handling

Exact request-rate assumptions must be configurable rather than hard-coded because access depends on the active Massive plan.

---

# 4. Statistical Models

Exactly three forecasting approaches are in scope.

## Model A — GBM Baseline

Geometric Brownian Motion provides the simplest stochastic benchmark.

Its purpose is comparison, not expected superiority.

---

## Model B — Markov Regime Model

Two regimes:

* lower-volatility
* higher-volatility

Regime numbers have no permanent semantic meaning.

After **every fit**, regimes are reordered based on fitted variance so that downstream code sees stable semantic labels.

---

## Model C — News-Markov

Uses Massive's news/sentiment information as an exogenous signal affecting Markov transition probabilities.

No LLM-generated feature enters the statistical forecasting path.

The question being tested is:

> Does recent financial-news information improve five-trading-day probabilistic forecasting beyond price-driven regime switching alone?

The project does not assume the answer is yes.

---

# 5. Known Statistical Implementation Constraints

These are mandatory implementation rules.

## Transition matrix orientation

statsmodels represents its transition matrix as:

```text
P[next regime = i | previous regime = j]
```

Therefore:

```text
rows    = next regime
columns = previous regime
```

and **columns sum to 1**.

Any Monte Carlo regime sampler must respect this orientation.

A unit test must verify:

```text
sum(P[:, j]) == 1
```

for every previous-state column.

A stationary-distribution test must also verify that the chosen matrix operations reproduce the expected stationary probabilities.

---

## TVTP temporal alignment

statsmodels defines transition matrix slice (t) as the transition:

```text
state at t-1
      ↓
state at t
```

and constructs that slice using the corresponding `exog_tvtp` row.

Therefore, if news observed on trading day (t) is intended to affect the transition:

```text
t → t+1
```

the TVTP input must be shifted/lag-aligned accordingly.

A no-lookahead unit test is mandatory.

---

## Filtered probabilities only

Historical forecasts must use:

```text
filtered_marginal_probabilities
```

not smoothed probabilities.

Filtered probabilities use observations available through time (t); smoothed probabilities incorporate later observations from the full sample.

Smoothed probabilities may be used for retrospective visualization only, never for backtest forecasting state.

---

## Regime re-sorting

Every successful Markov fit:

1. estimate regimes
2. inspect regime variance
3. reorder regimes
4. map lower variance → low-volatility state
5. map higher variance → high-volatility state
6. reorder dependent parameters/matrices consistently

Never assume statsmodels regime `0` is the calm state.

---

## Convergence fallback ladder

Forecast generation uses:

```text
Model C
News-Markov
   ↓ failure/non-convergence
Model B
Markov
   ↓ failure/non-convergence
Model A
GBM
```

Record which model actually generated every forecast.

A failed complex model must not crash the application.

---

# 6. News Data Constraints

Massive represents an article with a unique article ID and may associate an article with multiple tickers.

The normalized news table therefore uses:

```text
(article_id, ticker)
```

as its logical composite key.

This preserves ticker-specific sentiment/context for multi-company articles.

---

## Weekend and holiday news

News published while the market is closed is assigned to the **next valid trading day** for daily feature construction.

Example:

```text
Saturday article
Sunday article
        ↓
Monday news feature
```

More generally:

```text
publication timestamp
        ↓
next trading session
```

A market calendar, not naive weekday arithmetic, should determine the next session.

---

# 7. Feature Engineering

`daily_features` is the contract between Spark and the modeling layer.

Initial feature scope remains deliberately small.

Examples:

* close
* log return
* 5-day return
* rolling realized volatility
* short momentum feature
* volume normalization feature
* daily news sentiment
* short rolling news sentiment
* article count

Avoid feature expansion unless the existing model is complete and validated.

---

# 8. Monte Carlo

Forecast horizon:

**5 trading days**

Simulation count:

**5,000 paths**

Outputs should emphasize the distribution, not a point prediction.

Example outputs:

* median five-day return
* P10 return
* P90 return
* probability of positive return
* probability of decline beyond threshold
* current low/high-volatility regime probability

Raw 5,000-path arrays do not need to be permanently stored in Gold.

---

# 9. Walk-Forward Backtest

This is a core deliverable, not a stretch goal.

For each historical forecast origin (T):

```text
data available ≤ T
        ↓
fit
        ↓
forecast T+5
        ↓
observe actual T+5
        ↓
score
```

No future price or news observation may enter:

* model fitting
* regime probabilities
* features
* transition variables
* initialization

---

## Evaluation

Report pooled results across historical forecast origins.

Core measures:

* Brier score / probabilistic calibration
* median five-day return error
* prediction interval coverage

Every metric display must include:

```text
n = number of evaluated forecasts
```

The system must be able to truthfully show:

> No meaningful improvement detected at this sample size.

Model C is not presumed to win.

---

# 10. AI Search

Financial-news text is the project's required unstructured-data path.

```text
Silver news
    ↓
embeddings / AI Search
    ↓
hybrid retrieval
    ↓
Research Agent
```

The agent retrieves evidence relevant to the user's question and the selected ticker.

The retrieval system does not generate the numerical forecast.

---

# 11. Research Agent

Exactly one agent.

Exactly four core tools:

```text
get_market_forecast()

search_market_news()

update_watchlist()

save_research_report()
```

Read tools provide numerical/retrieval context.

Write tools modify Lakebase state.

The agent:

* retrieves
* synthesizes
* explains
* persists user actions

The agent does **not** calculate Markov models or Monte Carlo forecasts.

Architectural rule:

```text
Python / statistical system
        =
numerical inference

LLM
        =
reasoning + orchestration + explanation
```

---

# 12. Model Access

There is no routing subsystem.

All model calls go through one lightweight abstraction:

```text
call_model(task, ...)
```

Configuration determines the underlying Databricks model.

Application code must not scatter hard-coded model endpoint names throughout the repository.

This preserves replaceability without introducing:

* model tiers
* semantic routing
* AI Gateway
* escalation graphs
* routing benchmarks
* middleware infrastructure

---

# 13. Lightweight Model Telemetry

Every model call should record, when available:

* timestamp
* task
* model
* latency
* success/failure
* input token count
* output token count

This is instrumentation, **not a separate observability project**.

No dedicated AI-routing dashboard is required.

---

# 14. Streamlit Application

Three conceptual pages:

## Market Research

Shows:

* price history
* current regime
* five-day probability distribution
* forecast statistics
* relevant financial news

## Research Agent

Lets the user:

* ask questions about the forecast
* retrieve supporting news
* save reports
* update the watchlist

## Model Evaluation

Shows actual empirical comparison:

```text
GBM
vs
Markov
vs
News-Markov
```

including sample size.

No visualization may imply Model C wins before the backtest demonstrates it.

---

# 15. Reproducibility

The repository includes setup/rebuild assets.

```text
setup/
├── create_catalog.sql
├── create_delta_tables.sql
├── create_lakebase.sql
├── create_ai_search.py
├── create_workflow.py
└── seed_demo_data.py
```

Names may change during implementation.

The architectural requirement is that the environment can be recreated from code rather than depending entirely on manually configured workspace state.

---

# 16. Optional SLM Stretch Goal

Only after Checkpoints A, B and C work end-to-end.

A small/efficient model may classify news articles into event types such as:

```text
earnings
guidance
regulatory
product
management
M&A
macro
other
```

This output may be used for:

* UI labels
* retrieval metadata
* AI Search filtering

It **must not enter Model C**.

Failure to complete this feature has zero impact on core project completion.

---

# 17. Explicitly Out of Scope

Do not reintroduce:

* model router
* model tier architecture
* Unity AI Gateway
* dynamic model escalation
* Model D
* LLM-generated forecasting features
* Delta → Lakebase serving synchronization
* multi-agent orchestration
* custom AI-evaluation platform
* real-time/WebSocket market architecture
* minute-level forecasting
* portfolio optimization
* automated trading
* brokerage execution
* reinforcement learning
* LSTM/Transformer price forecasting
* GARCH expansion unless explicitly authorized later
* Heston/stochastic-volatility expansion
* knowledge graph
* unnecessary MCP architecture
* microservices
* Kubernetes
* custom GPU model infrastructure

Knowledge that these technologies exist is not permission to add them.

---

# 18. Build Checkpoints and Dates

Architecture work ends **Sunday, August 9, 2026**.

Implementation begins **Monday, August 10, 2026**.

## Checkpoint A — Data Works

### Monday, August 10

Goal:

```text
Massive API
→ Bronze
→ Silver
→ daily_features
```

Required demo:

* successful authenticated Massive request
* visible JSON response
* price ingestion
* news ingestion
* Bronze tables
* Silver tables
* daily features generated in Spark

**Checkpoint A freezes at end of day.**

Once accepted, do not reopen basic ingestion design unless a correctness bug requires it.

---

## Checkpoint B — Math Works

### Tuesday, August 11 → Thursday, August 13

This receives the largest time allocation.

Required:

* GBM baseline
* two-regime Markov model
* regime re-sorting
* transition-orientation tests
* News-Markov with correct TVTP alignment
* filtered-probability backtesting
* no-lookahead tests
* convergence fallback
* Monte Carlo
* Gold output
* walk-forward evaluation
* pooled metrics
* evaluation sample size

Required demo state:

```text
ticker
→ three models
→ five-day forecasts
→ empirical comparison
```

**Checkpoint B freezes Thursday.**

Once accepted, do not “improve the model” while building the application.

Only correctness bugs reopen B.

---

## Checkpoint C — Product Works

### Friday, August 14 → Saturday, August 15

Required:

* AI Search
* four-tool research agent
* Lakebase
* watchlist writes
* report persistence
* Lakebase CDF into Delta
* Streamlit application
* three usable pages
* end-to-end demo

Required end-to-end sequence:

```text
ingest
→ transform
→ model
→ forecast
→ evaluate
→ retrieve news
→ agent explanation
→ application write
→ CDC
```

**Checkpoint C freezes Saturday.**

At this point the capstone is complete.

---

## Checkpoint D — Polish Only

### Sunday, August 16

Permitted work:

* UI cleanup
* chart improvements
* README
* architecture diagram
* screenshots
* demo script
* model telemetry presentation
* optional SLM article labeling

Not permitted:

* new forecasting model
* new infrastructure subsystem
* new architecture
* additional datastore
* additional agent

If Checkpoint C needs repair, D is used for repair instead of stretch work.

---

# 19. First Implementation Action

Before building any tables:

```text
Databricks notebook
        ↓
Massive REST request
        ↓
one ticker
        ↓
HTTP 200
        ↓
inspect returned JSON
```

Massive's REST API is HTTPS-based and authenticated with an API key, so this is a direct validation of API credentials plus Databricks outbound connectivity.

If this does not work, stop and diagnose the environment/API connection.

Do not build around a source that has not yet been proven reachable.

---

# 20. Architecture Freeze Rule

This architecture is final.

Implementation questions should now be resolved **inside these boundaries**.

An implementation difficulty is not automatically an architectural problem.

Architecture may be reopened only if one of the following occurs:

1. an essential Databricks capability is unavailable;
2. an API/data limitation makes a core model impossible;
3. a statistical assumption is proven invalid;
4. a security or correctness defect requires structural change;
5. the project owner explicitly authorizes reconsideration.

Otherwise:

**build, test and finish.**
