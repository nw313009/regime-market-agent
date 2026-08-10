"""Model B — two-regime Markov switching model (spec B-2).

Contract::

    def fit_markov(returns_pct, exog_tvtp=None) -> FitResult   # or raises FitError
    def sort_regimes(res) -> SortedParams

``SortedParams`` fields: ``mus[2]``, ``sigmas[2]`` (low-volatility regime first),
``P`` (2x2, LEFT-stochastic, sorted), ``filtered_current`` (2,), ``perm``, ``converged``,
``degenerate_flags``.

fit_markov
----------
``MarkovRegression(endog, k_regimes=2, trend="c", switching_variance=True, exog_tvtp=...)``,
fitted with ``search_reps`` around 20 so several random starts are tried. Wrap the fit in
try/except; treat optimizer failure or non-finite parameters as ``FitError``.

All-NaN fitted parameters almost always mean returns were not scaled to percent, or NaNs
were left in ``endog``.

Degeneracy checks, which count as ``FitError`` for the fallback ladder:

- ``abs(sigma1 / sigma0 - 1) < 0.05`` — the two regimes are indistinguishable.
- any diagonal transition probability > 0.995 or < 0.005 — absorbing/degenerate chain.

sort_regimes
------------
Regime numbering from statsmodels carries no semantic meaning, so never assume regime 0 is
the calm state. After every successful fit:

- ``perm = argsort(fitted sigmas)``; apply ``perm`` to ``mus`` and ``sigmas``.
- ``P_sorted = P[np.ix_(perm, perm)]`` — correct for a left-stochastic matrix, since it
  permutes next-state rows and previous-state columns consistently.
- Reorder the columns of ``res.filtered_marginal_probabilities`` by ``perm`` and take the
  LAST ROW as ``filtered_current``.

NEVER read ``smoothed_marginal_probabilities`` anywhere in ``src/models/``. Smoothed
probabilities use the full sample, including the future, and a backtest that reads them
looks wonderful and means nothing. A grep-level test enforces this.
"""
