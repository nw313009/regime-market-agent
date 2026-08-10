"""Statistical modeling layer — pandas and statsmodels only (spec Checkpoint B).

Nothing in this package may import pyspark. Callers cross the boundary once::

    pdf = (spark.table(f"{catalog}.silver.daily_features")
                .where(col("ticker") == t)
                .orderBy("trade_date")
                .toPandas())

Roughly 2.5k rows per ticker makes that trivially safe.

Numerical conventions (spec B-0), mandatory everywhere in this package:

- Estimate in PERCENT log returns: ``r_pct = 100 * log_return``. Markov-switching MLE
  converges far more reliably at that scale. Divide fitted mu and sigma by 100 before
  simulation.
- Drop warm-up NaNs before fitting.
- Refuse to fit with fewer than ``cfg.backtest.min_train_days`` observations.

Mandatory statistical constraints (architecture doc section 5):

- The statsmodels transition matrix is LEFT-stochastic: ``P[next=i | prev=j]``, so rows are
  the next regime, columns the previous regime, and COLUMNS sum to 1.
- ``exog_tvtp`` is lagged one trading day.
- Only ``filtered_marginal_probabilities`` are ever read. Smoothed probabilities are
  forbidden anywhere under ``src/models/`` — they incorporate future observations and leak.
- Regimes are re-sorted by fitted variance after every fit.
- The fallback ladder is C -> B -> A, and the model actually used is always recorded.

Exactly three models are in scope. Model C is not presumed to win.
"""
