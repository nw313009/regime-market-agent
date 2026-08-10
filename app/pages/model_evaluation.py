"""Model Evaluation page (spec A2, C-5).

A pooled comparison of GBM vs Markov vs News-Markov on Brier score, median-return MAE and
80% interval coverage, read from the backtest pooled summary.

Non-negotiable display rules:

- ALWAYS render n, the number of evaluated forecasts.
- ALWAYS render the fallback rate — how often Model C failed to converge.
- The verdict line is computed from the numbers, with three possible cases: better, worse,
  or indistinguishable at this n. "No meaningful improvement detected at this sample size"
  is a first-class outcome, not a failure.

No visualization may imply Model C wins before the backtest demonstrates it.
"""
