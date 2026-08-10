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

Non-convergence is NOT a ``FitError`` here. It is recorded on ``SortedParams.converged`` and
reported as the fallback rate on the Model Evaluation page (spec A2), so the ladder in B-5
decides what to do with it and the number stays visible either way.

sort_regimes
------------
Regime numbering from statsmodels carries no semantic meaning, so never assume regime 0 is
the calm state. After every successful fit:

- ``perm = argsort(fitted sigmas)``; apply ``perm`` to ``mus`` and ``sigmas``.
- ``P_sorted = P[np.ix_(perm, perm)]`` — correct for a left-stochastic matrix, since it
  permutes next-state rows and previous-state columns consistently.
- Reorder the columns of the FILTERED marginal probabilities by ``perm`` and take the LAST ROW
  as ``filtered_current``.

NEVER read the smoothed marginal probabilities anywhere in ``src/models/``. They use the full
sample, including the future, and a backtest that reads them looks wonderful and means nothing.
Their statsmodels attribute name is deliberately absent from every file in this package so the
grep test in ``tests/test_no_lookahead.py`` can be a strict literal search.

VERIFIED AGAINST THE INSTALLED PACKAGE (statsmodels 0.14.6, not memory):

- ``MarkovRegression.fit`` takes ``search_reps`` (default 0); 20 requests 20 random starts. Those
  starts come from the LEGACY GLOBAL ``np.random``, and ``fit`` has no seed argument — which is why
  every fit here runs inside :func:`deterministic_start_search`.
- ``model.param_names`` is ``['p[0->0]', 'p[1->0]', 'const[0]', 'const[1]', 'sigma2[0]',
  'sigma2[1]']`` for this specification, so the regime mean is ``const[k]`` and the fitted
  parameter is a VARIANCE, ``sigma2[k]`` — sigma is its square root. With ``exog_tvtp`` the
  transition names gain ``.tvtp{j}`` suffixes and the ``const``/``sigma2`` names are unchanged.
- ``res.params`` is an ndarray for ndarray ``endog`` and a Series for Series ``endog``;
  ``res.filtered_marginal_probabilities`` is correspondingly an ndarray or a DataFrame, shaped
  ``(nobs, k_regimes)``. Everything here goes through ``np.asarray`` for that reason.
- ``res.regime_transition`` is ``(k, k, 1)`` for a time-invariant fit and ``(k, k, nobs)`` for a
  TVTP fit, and its COLUMNS sum to 1 — confirmed empirically, which is the orientation the
  architecture doc fixes.
- There is NO ``res.converged``. Convergence lives in ``res.mle_retvals["converged"]``.
- A constant or NaN-bearing ``endog`` surfaces as ``numpy.linalg.LinAlgError`` from the fit, not
  as a quiet NaN result, so the try/except is what catches it.

THE RESULT SURFACE THIS MODULE READS is exactly ``res.params``, ``res.model.param_names``,
``res.model.k_regimes``, ``res.regime_transition``, ``res.filtered_marginal_probabilities`` and
``res.mle_retvals``. It is written down because :func:`sort_regimes` is tested against a
hand-built unsorted result, and that test is only honest if the surface it fakes is the surface
production reads.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models import MIN_TRAIN_DAYS, FitError, to_decimal, validated_returns

__all__ = [
    "DIAGONAL_MAX",
    "DIAGONAL_MIN",
    "K_REGIMES",
    "MODEL_NAME",
    "SEARCH_REPS",
    "SIGMA_RATIO_MIN_SEPARATION",
    "START_SEARCH_SEED",
    "SortedParams",
    "degeneracy_flags",
    "deterministic_start_search",
    "fit_markov",
    "sort_regimes",
]

log = logging.getLogger(__name__)

#: Recorded as ``model_used`` when the ladder lands on this rung.
MODEL_NAME = "markov"

#: Two regimes: calm and turbulent. Not a tunable — every downstream schema (``gold.regime_states``
#: has exactly ``prob_low_vol``/``prob_high_vol``) is written for two.
K_REGIMES = 2

#: A switching intercept and a switching variance, per spec B-2.
TREND = "c"
SWITCHING_VARIANCE = True

#: Random starts for the MLE. The likelihood is multi-modal; a single start finds a local
#: optimum and calls it a regime structure.
SEARCH_REPS = 20

#: Seed for those random starts. See :func:`deterministic_start_search`. A fixed default rather
#: than the config seed: a fit should not change because someone re-seeded the simulation.
START_SEARCH_SEED = 20260810

#: Degeneracy thresholds (spec B-2). Below the separation the two regimes are the same regime
#: wearing two hats; outside the diagonal bounds the chain never switches or never stays.
SIGMA_RATIO_MIN_SEPARATION = 0.05
DIAGONAL_MAX = 0.995
DIAGONAL_MIN = 0.005

#: statsmodels parameter names for this specification, verified against 0.14.6. ``sigma2`` is a
#: VARIANCE; the model reports variances and this module reports standard deviations.
MEAN_PARAM = "const[{k}]"
VARIANCE_PARAM = "sigma2[{k}]"


@dataclass(frozen=True, eq=False)
class SortedParams:
    """Regime parameters after the mandatory re-sort, low-volatility regime first.

    Both scales are carried deliberately. ``mus_pct``/``sigmas_pct`` are the estimation scale
    (spec B-0) and are what belongs in a fit report; ``mus``/``sigmas`` are decimals and are what
    ``monte_carlo.run_forecast`` draws from. Returning only one scale means every consumer
    reimplements the divide-by-100, and the one that forgets produces a forecast wrong by two
    orders of magnitude that still looks like a forecast.

    ``eq=False`` because two of the fields are arrays: an accidental ``==`` should be an identity
    check, not an ambiguous-truth-value exception raised from inside a comparison.
    """

    #: Regime means, percent scale (estimation).
    mus_pct: tuple[float, float]
    #: Regime standard deviations, percent scale (estimation). ``sigmas_pct[0] <= sigmas_pct[1]``.
    sigmas_pct: tuple[float, float]
    #: Regime means, decimal scale (simulation).
    mus: tuple[float, float]
    #: Regime standard deviations, decimal scale (simulation).
    sigmas: tuple[float, float]
    #: 2x2 LEFT-stochastic transition matrix, sorted. ``P[next, prev]``, so COLUMNS sum to 1.
    P: np.ndarray
    #: Filtered regime probabilities at the last observation, sorted. Filtered, never smoothed.
    filtered_current: np.ndarray
    #: ``argsort`` of the fitted sigmas: ``sorted[i] == unsorted[perm[i]]``. Kept because
    #: ``monte_carlo`` must apply the SAME permutation to every per-day TVTP matrix it rebuilds.
    perm: np.ndarray
    #: Whether the optimizer converged. Reported, not raised on — see the module docstring.
    converged: bool
    #: The B-2 degeneracy checks, by name. All ``False`` after a successful ``fit_markov``,
    #: which raises on any of them; populated when ``sort_regimes`` is called on a result that
    #: did not come through that gate.
    degenerate_flags: Mapping[str, bool]

    @property
    def prob_low_vol(self) -> float:
        """Current filtered probability of the calm regime — ``gold.regime_states`` (spec B-6)."""
        return float(self.filtered_current[0])

    @property
    def prob_high_vol(self) -> float:
        """Current filtered probability of the turbulent regime (spec B-6)."""
        return float(self.filtered_current[1])


@contextmanager
def deterministic_start_search(seed: int):
    """Make ``search_reps`` reproducible, without leaving the process reseeded.

    VERIFIED IN statsmodels 0.14.6: the random start search draws its variates from
    ``np.random.uniform`` — the LEGACY GLOBAL RandomState — and ``fit`` exposes no seed or
    ``random_state`` argument. So two identical calls in one process start from different points and
    can land on different local optima of a multi-modal likelihood. Left alone, that makes every
    fitted number irreproducible: the same backtest run twice would publish different Brier scores,
    ``gold.forecast_runs.seed`` would document only half of what produced the row, and the B-7
    corrupt-the-future tests could not assert a bit-identical summary because the baseline itself
    would not be stable.

    Seeding the global state is the only hook the library offers, so the previous state is captured
    and restored: this makes OUR fit deterministic without silently reseeding a caller's notebook.

    TODO: drop this if statsmodels ever accepts a Generator for the start search.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def degeneracy_flags(
    sigmas: Sequence[float], transition_diagonal: Sequence[float]
) -> dict[str, bool]:
    """The two B-2 degeneracy checks, as a pure function of the fitted quantities.

    ``sigma_ratio``: ``abs(sigmas[1] / sigmas[0] - 1) < 0.05``. Two regimes whose volatilities
    agree to within 5% are one regime, and every "regime probability" the app would then show is
    noise dressed as information.

    ``transition_diagonal``: any diagonal probability above ``DIAGONAL_MAX`` or below
    ``DIAGONAL_MIN``. The first is an absorbing state the chain can never leave; the second is a
    state it can never stay in, which is a label swap rather than a regime.

    Pass the sigmas in either scale — the check is a ratio — and pass the diagonal of the
    left-stochastic matrix, ``P[k, k]`` being "stay in k".
    """
    sigma_pair = np.asarray(sigmas, dtype=float)
    diagonal = np.asarray(transition_diagonal, dtype=float)

    if sigma_pair.shape != (K_REGIMES,) or diagonal.shape != (K_REGIMES,):
        raise ValueError(
            f"expected {K_REGIMES} sigmas and {K_REGIMES} diagonal probabilities, got "
            f"{sigma_pair.shape} and {diagonal.shape}"
        )
    if not np.all(sigma_pair > 0):
        raise ValueError(f"sigmas must be positive, got {sigma_pair.tolist()}")

    return {
        "sigma_ratio": bool(
            abs(sigma_pair[1] / sigma_pair[0] - 1.0) < SIGMA_RATIO_MIN_SEPARATION
        ),
        "transition_diagonal": bool(
            np.any(diagonal > DIAGONAL_MAX) or np.any(diagonal < DIAGONAL_MIN)
        ),
    }


def fit_markov(
    returns_pct,
    exog_tvtp=None,
    *,
    min_obs: int = MIN_TRAIN_DAYS,
    search_reps: int = SEARCH_REPS,
    search_seed: int = START_SEARCH_SEED,
):
    """Fit a two-regime Markov switching model on percent log returns.

    Returns the raw statsmodels results object — ``monte_carlo`` needs it to rebuild the
    time-varying transition matrix per horizon day (spec B-4), so this function must not reduce
    it to parameters. Call :func:`sort_regimes` on it for the sorted, rescaled view.

    ``exog_tvtp`` is Model C's territory (spec B-3): pass the ``(nobs, 2)`` array of
    ``[ones, news_sentiment_3d.shift(1)]``. It is validated here rather than trusted, because
    both ways of getting it wrong are silent — an unshifted column leaks tomorrow's news into
    today's transition, and a missing ones column drops the intercept the transition model needs.

    The fit is REPRODUCIBLE: see :func:`deterministic_start_search` for why that takes explicit
    work and what ``search_seed`` controls.

    Raises ``FitError`` on: too few observations or non-finite input (spec B-0), optimizer
    failure, non-finite fitted parameters, and either degeneracy check.
    """
    endog = validated_returns(returns_pct, min_obs=min_obs, model=MODEL_NAME)
    exog = None if exog_tvtp is None else _validated_exog_tvtp(exog_tvtp, endog.size)

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        model = MarkovRegression(
            endog,
            k_regimes=K_REGIMES,
            trend=TREND,
            switching_variance=SWITCHING_VARIANCE,
            exog_tvtp=exog,
        )
        with deterministic_start_search(search_seed):
            res = model.fit(search_reps=search_reps)
    except Exception as exc:  # optimizer failure surfaces as LinAlgError, among others
        raise FitError(f"{MODEL_NAME}: fit failed: {type(exc).__name__}: {exc}") from exc

    _, sigmas_pct = _regime_params(res)
    flags = degeneracy_flags(sigmas_pct, _mean_transition_diagonal(res))
    failed = [name for name, tripped in flags.items() if tripped]
    if failed:
        raise FitError(
            f"{MODEL_NAME}: degenerate fit ({', '.join(failed)}): "
            f"sigmas_pct={np.round(sigmas_pct, 4).tolist()}, "
            f"mean diagonal={np.round(_mean_transition_diagonal(res), 4).tolist()}"
        )

    log.info(
        "%s fit n=%d tvtp=%s converged=%s sigmas_pct=%s",
        MODEL_NAME,
        endog.size,
        exog is not None,
        _converged(res),
        np.round(sigmas_pct, 4).tolist(),
    )
    return res


def sort_regimes(res) -> SortedParams:
    """Re-sort a fitted result so the low-volatility regime is index 0 (architecture doc §5).

    Every dependent quantity is permuted by the same ``perm``: the means, the sigmas, both rows
    and columns of the transition matrix via ``P[np.ix_(perm, perm)]``, and the columns of the
    filtered probabilities. Permuting some of them is worse than permuting none, because the
    result still looks well-formed.
    """
    mus_pct, sigmas_pct = _regime_params(res)
    perm = np.argsort(sigmas_pct, kind="stable")

    transitions = _transitions(res)
    # Last slice: for a time-invariant fit it is the only one; for a TVTP fit it is the
    # transition under the most recent news, and simulation rebuilds its own per-day matrices
    # anyway (spec B-4), so this one is a report of current conditions rather than an input.
    P = transitions[:, :, -1][np.ix_(perm, perm)]

    mus_sorted = mus_pct[perm]
    sigmas_sorted = sigmas_pct[perm]
    filtered_current = _filtered_current(res)[perm]

    return SortedParams(
        mus_pct=(float(mus_sorted[0]), float(mus_sorted[1])),
        sigmas_pct=(float(sigmas_sorted[0]), float(sigmas_sorted[1])),
        mus=(float(to_decimal(mus_sorted[0])), float(to_decimal(mus_sorted[1]))),
        sigmas=(float(to_decimal(sigmas_sorted[0])), float(to_decimal(sigmas_sorted[1]))),
        P=P,
        filtered_current=filtered_current,
        perm=perm,
        converged=_converged(res),
        degenerate_flags=degeneracy_flags(
            sigmas_sorted, _mean_transition_diagonal(res)[perm]
        ),
    )


# ------------------------------------------------------------------ result readers
# Everything below reads the documented result surface and nothing else. Each reader validates
# what it reads: a statsmodels release that renames a parameter or transposes a matrix must fail
# loudly here, not quietly produce a plausible forecast.


def _validated_exog_tvtp(exog_tvtp, nobs: int) -> np.ndarray:
    """Check the Model C transition input: 2-D, aligned with ``endog``, finite, ones first."""
    exog = np.asarray(exog_tvtp, dtype=float)
    if exog.ndim != 2:
        raise FitError(f"{MODEL_NAME}: exog_tvtp must be 2-D, got shape {exog.shape}")
    if exog.shape[0] != nobs:
        raise FitError(
            f"{MODEL_NAME}: exog_tvtp has {exog.shape[0]} rows for {nobs} endog observations — "
            "the shift(1) row was not dropped jointly with endog (spec B-3)"
        )
    if not np.all(np.isfinite(exog)):
        raise FitError(f"{MODEL_NAME}: exog_tvtp contains non-finite values")
    if not np.allclose(exog[:, 0], 1.0):
        raise FitError(
            f"{MODEL_NAME}: exog_tvtp column 0 must be the mandatory ones intercept (spec B-3)"
        )
    return exog


def _named_params(res) -> dict[str, float]:
    """``res.params`` keyed by ``res.model.param_names``, all values finite."""
    values = np.asarray(res.params, dtype=float).ravel()
    names = [str(name) for name in res.model.param_names]
    if len(names) != values.size:
        raise FitError(
            f"{MODEL_NAME}: {values.size} fitted parameters for {len(names)} names — "
            "unexpected statsmodels result shape"
        )
    if not np.all(np.isfinite(values)):
        raise FitError(
            f"{MODEL_NAME}: fitted parameters are not finite — returns are usually not scaled "
            "to percent, or NaNs were left in endog (spec B-0)"
        )
    return dict(zip(names, values.tolist()))


def _regime_params(res) -> tuple[np.ndarray, np.ndarray]:
    """Unsorted ``(mus_pct, sigmas_pct)``. statsmodels fits variances; this returns sigmas."""
    params = _named_params(res)
    k_regimes = int(getattr(res.model, "k_regimes", K_REGIMES))
    if k_regimes != K_REGIMES:
        raise FitError(f"{MODEL_NAME}: expected {K_REGIMES} regimes, model has {k_regimes}")

    try:
        mus = np.array([params[MEAN_PARAM.format(k=k)] for k in range(K_REGIMES)], dtype=float)
        variances = np.array(
            [params[VARIANCE_PARAM.format(k=k)] for k in range(K_REGIMES)], dtype=float
        )
    except KeyError as exc:
        raise FitError(
            f"{MODEL_NAME}: parameter {exc} missing from {sorted(params)} — statsmodels renamed "
            "a parameter for this specification"
        ) from exc

    if not np.all(variances > 0):
        raise FitError(f"{MODEL_NAME}: fitted variances must be positive, got {variances.tolist()}")
    return mus, np.sqrt(variances)


def _transitions(res) -> np.ndarray:
    """``(k, k, T)`` transition matrices, checked to be left-stochastic.

    ``T`` is 1 for a time-invariant fit and ``nobs`` for a TVTP fit. The column-sum assertion is
    the orientation guard: if a future statsmodels ever returns ``P[prev, next]`` instead, every
    forecast in this repo would be quietly transposed, so it fails here instead.
    """
    transitions = np.asarray(res.regime_transition, dtype=float)
    if transitions.ndim != 3 or transitions.shape[:2] != (K_REGIMES, K_REGIMES):
        raise FitError(
            f"{MODEL_NAME}: expected a ({K_REGIMES}, {K_REGIMES}, T) transition array, got shape "
            f"{transitions.shape}"
        )
    if not np.all(np.isfinite(transitions)):
        raise FitError(f"{MODEL_NAME}: transition matrix contains non-finite values")
    if not np.allclose(transitions.sum(axis=0), 1.0):
        raise FitError(
            f"{MODEL_NAME}: transition matrix is not LEFT-stochastic — columns must sum to 1 "
            "(architecture doc §5)"
        )
    return transitions


def _mean_transition_diagonal(res) -> np.ndarray:
    """Time-averaged ``P[k, k]``, the input to the diagonal degeneracy check.

    Averaging matters only for a TVTP fit, where the diagonal moves with the news: one unusual
    day pushing ``P[k, k]`` past 0.995 is not an absorbing chain, and treating it as one would
    fall back to Model B and inflate the fallback rate the Model Evaluation page reports. For a
    time-invariant fit there is a single slice and this is exactly ``np.diag(P)``.
    """
    transitions = _transitions(res)
    return np.diagonal(transitions, axis1=0, axis2=1).mean(axis=0)


def _filtered_current(res) -> np.ndarray:
    """Last row of the FILTERED marginal probabilities: the current regime distribution."""
    filtered = np.asarray(res.filtered_marginal_probabilities, dtype=float)
    if filtered.ndim != 2 or filtered.shape[1] != K_REGIMES:
        raise FitError(
            f"{MODEL_NAME}: expected filtered probabilities shaped (nobs, {K_REGIMES}), got "
            f"{filtered.shape}"
        )
    if filtered.shape[0] == 0:
        raise FitError(f"{MODEL_NAME}: filtered probabilities are empty")

    current = filtered[-1, :]
    if not np.all(np.isfinite(current)) or not np.isclose(current.sum(), 1.0):
        raise FitError(
            f"{MODEL_NAME}: current filtered probabilities do not sum to 1: {current.tolist()}"
        )
    return current


def _converged(res) -> bool:
    """``res.mle_retvals["converged"]``. There is no ``res.converged`` in statsmodels 0.14.6."""
    retvals: Any = getattr(res, "mle_retvals", None)
    if not isinstance(retvals, Mapping):
        return False
    return bool(retvals.get("converged", False))
