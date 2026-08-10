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
- Only filtered marginal probabilities are ever read. Smoothed probabilities are forbidden
  anywhere under ``src/models/`` — they incorporate future observations and leak. The
  identifier itself is deliberately absent from every file in this package so the grep test
  in ``tests/test_no_lookahead.py`` can be a strict literal search.
- Regimes are re-sorted by fitted variance after every fit.
- The fallback ladder is C -> B -> A, and the model actually used is always recorded.

Exactly three models are in scope. Model C is not presumed to win.

WHAT THIS PACKAGE MODULE OWNS. The B-0 conventions above are executable here rather than
restated in each model: :func:`percent_log_returns` and :func:`percent_returns` build the
estimation input, :func:`to_decimal` performs the one rescale that separates estimation from
simulation, :func:`validated_returns` is the shared refusal gate, and :class:`FitError` is the
single exception type the fallback ladder catches. The frozen repo layout (spec B0) offers no
other module where all five models may share code, and ``src/pipelines/__init__.py`` already
sets the precedent of a package module holding what its siblings share.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "MIN_TRAIN_DAYS",
    "PERCENT_SCALE",
    "FitError",
    "percent_log_returns",
    "percent_returns",
    "to_decimal",
    "validated_returns",
    "warmup_offset",
]


class FitError(RuntimeError):
    """A model refused to fit, or fitted something unusable (spec B-2).

    One exception type for the whole modeling layer, because the fallback ladder needs exactly
    one thing to catch: ``C -> B -> A`` descends on a ``FitError`` from the rung above, records
    the reason, and never lets a failed complex model crash the application (architecture doc
    section 5).

    Raised for: too few observations, non-finite input, optimizer failure, non-finite fitted
    parameters, and the B-2 degeneracy checks. NOT raised for non-convergence, which is
    recorded on :class:`~src.models.markov.SortedParams` instead — see ``fit_markov``.
    """


#: The estimation scale (spec B-0). Percent log returns, i.e. ``100 * log_return``.
PERCENT_SCALE = 100.0

#: Minimum observations any model will fit on. Mirrors ``backtest.min_train_days`` in
#: config.yaml; a caller holding the config should pass its value explicitly (B-5 does) rather
#: than rely on this default staying in step.
MIN_TRAIN_DAYS = 252


def percent_log_returns(closes: Sequence[float] | Any) -> np.ndarray:
    """``100 * ln(close_t / close_{t-1})`` — the estimation input, from prices.

    The first row has no predecessor and is dropped, so the result is one element shorter than
    ``closes``. This is the same definition Spark writes into ``silver.daily_features.log_return``
    (spec A-4), scaled to percent; :func:`percent_returns` consumes that column instead and
    ``tests/test_models.py`` pins the two against each other.

    A non-positive or non-finite close is a data error rather than a warm-up gap, so it raises
    ``ValueError``: no model can repair it and the fallback ladder should not be asked to try.
    """
    prices = np.asarray(closes, dtype=float)
    if prices.ndim != 1:
        raise ValueError(f"closes must be 1-D, got shape {prices.shape}")
    if prices.size < 2:
        raise ValueError("at least two closes are needed to form one return")
    if not np.all(np.isfinite(prices)):
        raise ValueError("closes contain non-finite values")
    if np.any(prices <= 0):
        raise ValueError("closes must be strictly positive to take a log return")
    return PERCENT_SCALE * np.log(prices[1:] / prices[:-1])


def warmup_offset(log_returns: Sequence[float | None] | Any) -> int:
    """How many leading warm-up rows :func:`percent_returns` drops.

    Exposed because the estimation input is not the only thing sliced from those rows: Model C's
    ``exog_tvtp`` has to be trimmed by the SAME amount or the news is silently shifted against the
    returns, which is the lookahead bug the lag rule exists to prevent. One function decides the
    offset; every caller aligning a second column to the returns asks it.
    """
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"log_returns must be 1-D, got shape {values.shape}")

    usable = np.flatnonzero(~np.isnan(values))
    if usable.size == 0:
        raise ValueError("every log_return is NaN — nothing to estimate on")
    return int(usable[0])


def percent_returns(log_returns: Sequence[float | None] | Any) -> np.ndarray:
    """``silver.daily_features.log_return`` -> percent log returns, warm-up dropped (spec B-0).

    Only the LEADING run of NaNs is warm-up: ``log_return`` is NULL on a ticker's first row
    because there is no previous close (spec A-4). A NaN anywhere after that is a hole in the
    price history, and dropping it would splice two non-adjacent sessions into one step — which
    fabricates a transition in a model whose entire subject is the transition. So an interior
    NaN raises ``ValueError`` rather than being quietly removed, and it is deliberately not a
    ``FitError``: it breaks all three models identically, so descending the ladder would report
    "every model failed" instead of the actual data defect.
    """
    values = np.asarray(log_returns, dtype=float)
    offset = warmup_offset(values)

    trimmed = values[offset:]
    if not np.all(np.isfinite(trimmed)):
        holes = np.flatnonzero(~np.isfinite(trimmed)) + offset
        raise ValueError(
            f"log_return has {holes.size} non-finite value(s) after the warm-up rows, at "
            f"index/indices {holes[:5].tolist()} — that is a gap in the price history, not "
            "warm-up, and it must not be spliced out"
        )
    return PERCENT_SCALE * trimmed


def to_decimal(values: Any) -> Any:
    """Percent scale -> decimal scale: the single rescale mandated by spec B-0.

    Fitted mu and sigma are estimated in percent and simulated in decimals. Named rather than
    inlined so the divide-by-100 appears in one place and a missing rescale is a missing call
    instead of an invisible factor of 100.
    """
    if np.isscalar(values):
        return float(values) / PERCENT_SCALE
    return np.asarray(values, dtype=float) / PERCENT_SCALE


def validated_returns(returns_pct: Any, *, min_obs: int, model: str) -> np.ndarray:
    """The shared refusal gate (spec B-0): 1-D, finite, and long enough, or ``FitError``.

    ``model`` names the caller so the ladder's recorded reason says which rung refused.
    """
    returns = np.asarray(returns_pct, dtype=float)
    if returns.ndim != 1:
        raise FitError(f"{model}: returns_pct must be 1-D, got shape {returns.shape}")
    if not np.all(np.isfinite(returns)):
        raise FitError(
            f"{model}: returns_pct contains non-finite values — warm-up NaNs must be dropped "
            "before fitting (spec B-0)"
        )
    if returns.size < min_obs:
        raise FitError(
            f"{model}: refusing to fit on {returns.size} observations, {min_obs} required "
            "(spec B-0)"
        )
    return returns
