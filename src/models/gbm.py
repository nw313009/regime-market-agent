"""Model A — Geometric Brownian Motion baseline (spec B-1).

Contract::

    fit_gbm(returns_pct) -> {"mu": float, "sigma": float}

Parity rule: the fit uses the SAME training window the Markov models use at that origin.
A baseline fitted on a different window is not a baseline, it is a different experiment.

Simulation lives in ``monte_carlo.py``; GBM paths draw ``r ~ Normal(mu, sigma)`` with no
regimes.

Purpose is comparison, not expected inferiority.

SCALE. ``mu`` and ``sigma`` come out in the PERCENT scale they went in as, because that is the
estimation scale (spec B-0) and because returning a silently rescaled number under the same key
would make the two scales indistinguishable at the call site. The caller rescales for
simulation with :func:`src.models.to_decimal`, exactly as it does for the Markov parameters.
"""

from __future__ import annotations

import logging

import numpy as np

from src.models import MIN_TRAIN_DAYS, FitError, validated_returns

__all__ = ["MODEL_NAME", "fit_gbm"]

log = logging.getLogger(__name__)

#: Recorded as ``model_used`` when the ladder lands on this rung.
MODEL_NAME = "gbm"


def fit_gbm(returns_pct, min_obs: int = MIN_TRAIN_DAYS) -> dict[str, float]:
    """Fit the GBM baseline on percent log returns.

    ``sigma`` is the sample standard deviation with ``ddof=1``, matching
    ``realized_vol_20d``'s ``stddev_samp`` in ``silver.daily_features`` (spec A-4) so the
    baseline's volatility and the feature table's volatility mean the same thing.

    A zero-variance training window raises ``FitError``: the arithmetic succeeds but the
    resulting "forecast" is a constant, which is not a distribution and would be reported as
    though it were one. GBM is the last rung of the ladder, so this is the one place a refusal
    has nowhere to fall back to — and that is the point: it means the input was constant, and
    the run should say so rather than emit a degenerate forecast.
    """
    returns = validated_returns(returns_pct, min_obs=min_obs, model=MODEL_NAME)

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))

    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise FitError(f"{MODEL_NAME}: fitted mu/sigma are not finite")
    if sigma <= 0.0:
        raise FitError(
            f"{MODEL_NAME}: fitted sigma is {sigma} — the training window has no variance"
        )

    log.info("%s fit n=%d mu_pct=%.6f sigma_pct=%.6f", MODEL_NAME, returns.size, mu, sigma)
    return {"mu": mu, "sigma": sigma}
