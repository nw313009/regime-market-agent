"""Model A — Geometric Brownian Motion baseline (spec B-1).

Contract::

    fit_gbm(returns) -> {"mu": float, "sigma": float}

Parity rule: the fit uses the SAME training window the Markov models use at that origin.
A baseline fitted on a different window is not a baseline, it is a different experiment.

Simulation lives in ``monte_carlo.py``; GBM paths draw ``r ~ Normal(mu, sigma)`` with no
regimes.

Purpose is comparison, not expected inferiority.
"""
