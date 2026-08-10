"""Markov model tests (spec B-7). All mandatory before Checkpoint B freezes.

- ``test_transition_orientation``: for any fitted P, assert ``np.allclose(P.sum(axis=0), 1)``.
  The matrix is left-stochastic, so COLUMNS sum to 1.
- ``test_stationary``: solve for pi from P as the left eigenvector under the column
  convention, then simulate 200k steps with the production sampler and assert the empirical
  frequencies land within 1% of pi. This is the test that catches a transposed sampler
  instantly, and it is the reason it exists.
- ``test_fallback``: inject a ``FitError`` from Model C and assert Model B was used AND that
  the substitution was recorded. A silent fallback is as bad as a crash.
- Regime sorting: assert the low-volatility regime is always index 0 after
  ``sort_regimes``, and that ``mus``, ``sigmas``, ``P`` and the filtered probabilities were
  all permuted consistently.
- Degeneracy: assert the sigma-ratio and diagonal-probability checks raise ``FitError`` so
  the ladder actually descends.

HOW THE SORTING TEST IS BUILT. ``sort_regimes`` is exercised twice: on a HAND-BUILT result whose
regimes arrive in the wrong order, where every output can be checked against a value written down
by hand, and on the real fitted result, where the check is that the recovered ordering matches the
regime path that generated the data. The hand-built case is the one that pins the permutation
algebra; the fitted case is the one that proves the algebra is wired to statsmodels correctly.
Neither is sufficient alone.

THE B-5 SECTION covers the walk-forward machinery: which origins are eligible, what the shared
window contains, the ladder's descent and its recorded reasons, the three scoring rules, and the
pooling arithmetic. The scoring rules are asserted on hand-chosen numbers rather than on fitted
output, because a fit cannot be made to land exactly on an interval edge — and the edge is where a
coverage rule is most likely to be wrong. The corrupt-the-future half of B-5 lives in
``tests/test_no_lookahead.py``, where the leakage tests are.
"""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.models import (
    MIN_TRAIN_DAYS,
    PERCENT_SCALE,
    FitError,
    percent_log_returns,
    percent_returns,
    to_decimal,
)
from src.models import backtest
from src.models.backtest import (
    LADDERS,
    MODEL_ARMS,
    BacktestRow,
    feature_dates,
    fit_arm,
    origin_window,
    pooled_summary,
    run_backtest,
    score_forecast,
    weekly_origins,
)
from src.models.gbm import fit_gbm
from src.models.markov import (
    DIAGONAL_MAX,
    DIAGONAL_MIN,
    K_REGIMES,
    SIGMA_RATIO_MIN_SEPARATION,
    degeneracy_flags,
    fit_markov,
    sort_regimes,
)
from tests.conftest import TRUE_SIGMAS_PCT, backtest_config

MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"

#: statsmodels parameter order for ``MarkovRegression(k_regimes=2, trend="c",
#: switching_variance=True)``, verified against 0.14.6.
PARAM_NAMES = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2[0]", "sigma2[1]"]


def fake_result(
    *,
    variances: tuple[float, float],
    mus: tuple[float, float],
    P,
    filtered,
    converged: bool = True,
):
    """A minimal stand-in for a statsmodels result, covering exactly the surface we read.

    That surface is documented in ``src/models/markov.py``: ``params``, ``model.param_names``,
    ``model.k_regimes``, ``regime_transition``, ``filtered_marginal_probabilities`` and
    ``mle_retvals``. Faking it is the only way to feed ``sort_regimes`` regimes in a KNOWN wrong
    order — an optimizer cannot be asked to produce one on demand.

    The two transition entries inside ``params`` are placeholders: nothing reads the fitted
    ``p[i->j]`` parameters, because the transition probabilities are read from
    ``regime_transition``, which is where statsmodels has already applied the logistic transform.
    """
    params = np.array([0.9, 0.1, mus[0], mus[1], variances[0], variances[1]], dtype=float)
    return SimpleNamespace(
        params=params,
        model=SimpleNamespace(param_names=list(PARAM_NAMES), k_regimes=K_REGIMES, tvtp=False),
        regime_transition=np.asarray(P, dtype=float).reshape(K_REGIMES, K_REGIMES, 1),
        filtered_marginal_probabilities=np.asarray(filtered, dtype=float),
        mle_retvals={"converged": converged},
    )


# ------------------------------------------------------------------ B-0 estimation input


def test_log_returns_exact_values_on_synthetic_prices():
    """Hand-computable series, hand-written answers (spec B-7)."""
    assert percent_log_returns([100.0, 110.0, 99.0]) == pytest.approx(
        [100.0 * math.log(1.1), 100.0 * math.log(0.9)]
    )

    # Constant growth: every return is the same number, and there are n-1 of them.
    closes = 100.0 * 1.01 ** np.arange(6)
    returns = percent_log_returns(closes)
    assert returns.shape == (5,)
    assert returns == pytest.approx(np.full(5, 100.0 * math.log(1.01)))

    # The percent scale is exactly 100x, not "about 100x" (spec B-0).
    assert percent_log_returns([100.0, 100.0 * math.e]) == pytest.approx([PERCENT_SCALE])


def test_percent_returns_matches_the_spark_log_return_column():
    """The models' input and ``silver.daily_features.log_return`` must be the same number.

    A-4 writes ``ln(close / lag(close))`` with NULL on the first row. B-0 says the modeling layer
    scales it to percent and drops the warm-up. If these two paths ever disagree, every backtest
    number is quietly computed on a different series than the one the feature table shows.
    """
    closes = np.array([100.0, 101.5, 99.0, 103.25, 102.0])
    spark_column = [None, *np.log(closes[1:] / closes[:-1]).tolist()]

    assert percent_returns(spark_column) == pytest.approx(percent_log_returns(closes))


def test_percent_returns_drops_leading_warmup_only():
    log_return = [np.nan, np.nan, 0.01, -0.02]

    assert percent_returns(log_return) == pytest.approx([1.0, -2.0])


def test_percent_returns_refuses_an_interior_gap():
    """An interior NaN is a hole in the price history, and splicing it fabricates a transition."""
    with pytest.raises(ValueError, match="not"):
        percent_returns([np.nan, 0.01, np.nan, 0.02])


@pytest.mark.parametrize(
    "closes",
    [
        [100.0],  # one close cannot make a return
        [100.0, 0.0],  # a zero close has no log
        [100.0, -5.0],  # nor a negative one
        [100.0, np.nan],
    ],
)
def test_percent_log_returns_rejects_unusable_prices(closes):
    with pytest.raises(ValueError):
        percent_log_returns(closes)


def test_percent_returns_rejects_an_all_nan_column():
    with pytest.raises(ValueError, match="NaN"):
        percent_returns([np.nan, np.nan])


def test_to_decimal_is_the_only_rescale():
    assert to_decimal(3.0) == pytest.approx(0.03)
    assert to_decimal([1.0, 3.0]) == pytest.approx([0.01, 0.03])


# ------------------------------------------------------------------ B-0 refusal gate


@pytest.mark.parametrize("fit", [fit_gbm, fit_markov])
def test_models_refuse_too_short_a_window(fit):
    """Spec B-0: fewer than ``min_train_days`` observations is a refusal, not a small sample."""
    with pytest.raises(FitError, match="refusing to fit"):
        fit(np.zeros(MIN_TRAIN_DAYS - 1))


@pytest.mark.parametrize("fit", [fit_gbm, fit_markov])
def test_models_refuse_non_finite_returns(fit):
    returns = np.full(MIN_TRAIN_DAYS, 0.5)
    returns[7] = np.nan

    with pytest.raises(FitError, match="non-finite"):
        fit(returns)


def test_fit_gbm_refuses_a_constant_window():
    with pytest.raises(FitError, match="no variance"):
        fit_gbm(np.full(MIN_TRAIN_DAYS, 0.25))


def test_fit_gbm_recovers_known_moments(two_regime_returns_pct):
    """The baseline is the pooled mean and sample sigma of the SAME window (parity rule, B-1)."""
    fitted = fit_gbm(two_regime_returns_pct)

    assert fitted["mu"] == pytest.approx(float(np.mean(two_regime_returns_pct)))
    assert fitted["sigma"] == pytest.approx(float(np.std(two_regime_returns_pct, ddof=1)))
    # One volatility for a two-volatility world: it must land between the regimes, which is the
    # whole reason Model B might beat it.
    assert TRUE_SIGMAS_PCT[0] < fitted["sigma"] < TRUE_SIGMAS_PCT[1]


# ------------------------------------------------------------------ B-2 transition orientation


def test_transition_orientation(sorted_markov, fitted_markov):
    """COLUMNS sum to 1 (architecture doc §5). The one assertion B-7 spells out verbatim."""
    assert np.allclose(sorted_markov.P.sum(axis=0), 1)

    raw = np.asarray(fitted_markov.regime_transition, dtype=float)
    assert raw.shape == (K_REGIMES, K_REGIMES, 1)
    assert np.allclose(raw.sum(axis=0), 1)


def test_transition_orientation_is_not_symmetric_by_accident(sorted_markov):
    """A symmetric fitted P would let a transposed sampler pass the column-sum test."""
    assert not np.allclose(sorted_markov.P, sorted_markov.P.T)


# ------------------------------------------------------------------ B-2 sort_regimes


def test_sort_regimes_permutes_every_field_consistently():
    """Hand-built UNSORTED result, hand-written expectations."""
    # Regime 0 is the turbulent one here (variance 9 vs 1), so the sort must swap everything.
    raw_P = np.array([[0.80, 0.30], [0.20, 0.70]])
    filtered = np.array([[0.50, 0.50], [0.25, 0.75]])
    res = fake_result(variances=(9.0, 1.0), mus=(-0.5, 0.2), P=raw_P, filtered=filtered)

    sorted_params = sort_regimes(res)

    assert sorted_params.perm.tolist() == [1, 0]
    assert sorted_params.sigmas_pct == pytest.approx((1.0, 3.0))
    assert sorted_params.mus_pct == pytest.approx((0.2, -0.5))
    # Decimal scale is the percent scale over 100, for simulation (spec B-0).
    assert sorted_params.sigmas == pytest.approx((0.01, 0.03))
    assert sorted_params.mus == pytest.approx((0.002, -0.005))
    # P[np.ix_(perm, perm)]: next-state rows and previous-state columns move together, so the
    # calm regime's staying probability is the old P[1, 1].
    assert sorted_params.P == pytest.approx(np.array([[0.70, 0.20], [0.30, 0.80]]))
    assert np.allclose(sorted_params.P.sum(axis=0), 1)
    # Columns of the filtered probabilities reorder; the LAST ROW is the current distribution.
    assert sorted_params.filtered_current == pytest.approx([0.75, 0.25])
    assert sorted_params.prob_low_vol == pytest.approx(0.75)
    assert sorted_params.prob_high_vol == pytest.approx(0.25)
    assert sorted_params.converged is True
    assert sorted_params.degenerate_flags == {"sigma_ratio": False, "transition_diagonal": False}


def test_sort_regimes_is_the_identity_when_already_sorted():
    raw_P = np.array([[0.80, 0.30], [0.20, 0.70]])
    filtered = np.array([[0.50, 0.50], [0.25, 0.75]])
    res = fake_result(variances=(1.0, 9.0), mus=(0.2, -0.5), P=raw_P, filtered=filtered)

    sorted_params = sort_regimes(res)

    assert sorted_params.perm.tolist() == [0, 1]
    assert sorted_params.sigmas_pct == pytest.approx((1.0, 3.0))
    assert sorted_params.mus_pct == pytest.approx((0.2, -0.5))
    assert sorted_params.P == pytest.approx(raw_P)
    assert sorted_params.filtered_current == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize("variances", [(9.0, 1.0), (1.0, 9.0)])
def test_sort_regimes_permutation_holds_elementwise(variances):
    """The permutation identity itself: ``sorted[i] == unsorted[perm[i]]`` for every field."""
    raw_P = np.array([[0.91, 0.17], [0.09, 0.83]])
    filtered = np.array([[0.40, 0.60], [0.30, 0.70]])
    raw_mus = (0.11, -0.42)
    res = fake_result(variances=variances, mus=raw_mus, P=raw_P, filtered=filtered)

    sorted_params = sort_regimes(res)
    perm = sorted_params.perm
    raw_sigmas = np.sqrt(np.asarray(variances))

    for i in range(K_REGIMES):
        assert sorted_params.sigmas_pct[i] == pytest.approx(raw_sigmas[perm[i]])
        assert sorted_params.mus_pct[i] == pytest.approx(raw_mus[perm[i]])
        assert sorted_params.filtered_current[i] == pytest.approx(filtered[-1, perm[i]])
        for j in range(K_REGIMES):
            assert sorted_params.P[i, j] == pytest.approx(raw_P[perm[i], perm[j]])

    assert sorted_params.sigmas_pct[0] <= sorted_params.sigmas_pct[1]


def test_sort_regimes_records_non_convergence_without_raising():
    """Non-convergence is recorded, not raised on: the fallback rate is a reported number."""
    res = fake_result(
        variances=(1.0, 9.0),
        mus=(0.0, 0.0),
        P=np.array([[0.9, 0.2], [0.1, 0.8]]),
        filtered=np.array([[0.5, 0.5]]),
        converged=False,
    )

    assert sort_regimes(res).converged is False


def test_sort_regimes_rejects_a_right_stochastic_matrix():
    """The orientation guard: rows summing to 1 instead of columns must fail loudly."""
    row_stochastic = np.array([[0.90, 0.10], [0.30, 0.70]])
    res = fake_result(
        variances=(1.0, 9.0),
        mus=(0.0, 0.0),
        P=row_stochastic,
        filtered=np.array([[0.5, 0.5]]),
    )

    with pytest.raises(FitError, match="LEFT-stochastic"):
        sort_regimes(res)


# ------------------------------------------------------------------ B-2 degeneracy


def test_degeneracy_flags_healthy_fit():
    assert degeneracy_flags((1.0, 3.0), (0.95, 0.90)) == {
        "sigma_ratio": False,
        "transition_diagonal": False,
    }


def test_degeneracy_flags_indistinguishable_regimes():
    """Two regimes agreeing to within 5% on volatility are one regime with two labels."""
    flags = degeneracy_flags((1.0, 1.0 + SIGMA_RATIO_MIN_SEPARATION / 2), (0.95, 0.90))

    assert flags["sigma_ratio"] is True
    assert flags["transition_diagonal"] is False


@pytest.mark.parametrize(
    "diagonal",
    [
        (DIAGONAL_MAX + 0.001, 0.90),  # absorbing: it never leaves
        (0.95, DIAGONAL_MIN - 0.001),  # never stays: a label swap, not a regime
    ],
)
def test_degeneracy_flags_degenerate_chain(diagonal):
    flags = degeneracy_flags((1.0, 3.0), diagonal)

    assert flags["transition_diagonal"] is True
    assert flags["sigma_ratio"] is False


def test_degeneracy_thresholds_are_exclusive_at_the_boundary():
    """The spec's bounds are strict: exactly on the threshold is not degenerate."""
    flags = degeneracy_flags((1.0, 1.0 + SIGMA_RATIO_MIN_SEPARATION), (DIAGONAL_MAX, DIAGONAL_MIN))

    assert flags == {"sigma_ratio": False, "transition_diagonal": False}


def one_switch_series(seed: int, length: int = 600):
    """Calm for the first half, turbulent for the second, and never back: an absorbing chain."""
    rng = np.random.default_rng(seed)
    half = length // 2
    return np.concatenate(
        [rng.normal(0.0, TRUE_SIGMAS_PCT[0], half), rng.normal(0.0, TRUE_SIGMAS_PCT[1], half)]
    )


def alternating_series(seed: int, length: int = 400):
    """Volatility flipping every single day: a chain that can never stay where it is."""
    rng = np.random.default_rng(seed)
    sigmas = np.where(np.arange(length) % 2 == 0, TRUE_SIGMAS_PCT[0], TRUE_SIGMAS_PCT[1])
    return rng.normal(0.0, sigmas)


@pytest.mark.parametrize("build", [one_switch_series, alternating_series], ids=["absorbing", "never_stays"])
@pytest.mark.parametrize("seed", [4, 42])
def test_fit_markov_raises_fit_error_on_a_degenerate_chain(build, seed):
    """Both diagonal failure modes must descend the ladder rather than publish a forecast.

    A window containing exactly one regime change fits a diagonal of about 0.998 — the chain
    never leaves where it is, so a five-day transition forecast from it is a straight line with
    error bars. A window that flips every day fits a diagonal of 0.0 — the "regimes" are a
    relabelling of alternate days. Both are ``FitError`` by spec B-2, and both are checked at two
    seeds because a degeneracy check that only fires on one draw is not a check.
    """
    with pytest.raises(FitError, match="degenerate fit"):
        fit_markov(build(seed))


def test_sort_regimes_reports_degeneracy_it_did_not_raise_on():
    """``fit_markov`` gates on degeneracy; ``sort_regimes`` reports it for anything else."""
    res = fake_result(
        variances=(1.0, 1.0),
        mus=(0.0, 0.0),
        P=np.array([[0.999, 0.2], [0.001, 0.8]]),
        filtered=np.array([[0.5, 0.5]]),
    )

    assert sort_regimes(res).degenerate_flags == {
        "sigma_ratio": True,
        "transition_diagonal": True,
    }


# ------------------------------------------------------------------ B-2 exog_tvtp validation


def test_fit_markov_rejects_misaligned_exog_tvtp():
    """The C-e failure mode: the ``shift(1)`` row was not dropped jointly with ``endog``."""
    returns = np.random.default_rng(5).normal(0.0, 1.0, MIN_TRAIN_DAYS)
    exog = np.column_stack([np.ones(MIN_TRAIN_DAYS + 1), np.zeros(MIN_TRAIN_DAYS + 1)])

    with pytest.raises(FitError, match=r"shift\(1\)"):
        fit_markov(returns, exog_tvtp=exog)


def test_fit_markov_requires_the_ones_intercept_column():
    """The column of ones is mandatory (spec B-3): it is the transition model's intercept."""
    returns = np.random.default_rng(6).normal(0.0, 1.0, MIN_TRAIN_DAYS)
    exog = np.column_stack([np.zeros(MIN_TRAIN_DAYS), np.zeros(MIN_TRAIN_DAYS)])

    with pytest.raises(FitError, match="ones intercept"):
        fit_markov(returns, exog_tvtp=exog)


# ------------------------------------------------------------------ end-to-end recovery


def test_fit_markov_recovers_the_known_two_regime_structure(sorted_markov):
    """Fit -> sort on N(0, 1%) / N(0, 3%) segments: the ordering and the scales must be right.

    This is the integration test for B-1/B-2: real statsmodels, real MLE, real permutation, and a
    truth to compare against. The tolerances are loose because MLE on 800 observations is not
    exact; the ORDERING is not loose, because a swapped ordering is the failure this whole
    re-sorting rule exists to prevent.
    """
    low, high = sorted_markov.sigmas_pct

    assert low < high
    assert low == pytest.approx(TRUE_SIGMAS_PCT[0], rel=0.25)
    assert high == pytest.approx(TRUE_SIGMAS_PCT[1], rel=0.25)

    # Decimal scale is what monte_carlo draws from, and it is the percent scale over 100.
    assert sorted_markov.sigmas == pytest.approx(to_decimal(np.asarray(sorted_markov.sigmas_pct)))
    assert sorted_markov.mus == pytest.approx(to_decimal(np.asarray(sorted_markov.mus_pct)))

    assert sorted_markov.converged is True
    assert not any(sorted_markov.degenerate_flags.values())

    # Both regimes are persistent, which is what makes them regimes rather than outliers.
    assert np.all(np.diag(sorted_markov.P) > 0.5)

    current = sorted_markov.filtered_current
    assert current.shape == (K_REGIMES,)
    assert current.sum() == pytest.approx(1.0)
    assert np.all((current >= 0.0) & (current <= 1.0))


def test_sorted_regimes_agree_with_the_true_regime_path(fitted_markov, sorted_markov, true_regime_states):
    """The sorted labels must mean what they claim: index 0 IS the calm regime.

    Classifying each day by the argmax of its sorted filtered probabilities should mostly recover
    the simulated regime path. If the permutation were applied inconsistently the accuracy would
    collapse toward its complement, which no tolerance on the sigmas would catch.
    """
    filtered = np.asarray(fitted_markov.filtered_marginal_probabilities, dtype=float)
    predicted = np.argmax(filtered[:, sorted_markov.perm], axis=1)

    accuracy = float(np.mean(predicted == true_regime_states))
    assert accuracy > 0.8


# ------------------------------------------------------------------ fit reproducibility


def test_two_identical_fits_return_identical_parameters(two_regime_returns_pct):
    """The same window must always give the same fit — exactly, not approximately.

    statsmodels draws its ``search_reps`` random starts from the legacy global ``np.random`` and
    offers no seed argument, so without the pinning in ``deterministic_start_search`` two calls in
    one process start from different points and can land on different local optima. Everything
    downstream depends on this: a backtest run twice would publish different Brier scores, and the
    corrupt-the-future tests could not demand a bit-identical summary.
    """
    window = np.asarray(two_regime_returns_pct)[:300]

    first = np.asarray(fit_markov(window).params, dtype=float)
    second = np.asarray(fit_markov(window).params, dtype=float)

    assert np.array_equal(first, second)


def test_fitting_leaves_the_global_random_state_untouched(two_regime_returns_pct):
    """Pinning the start search must not reseed the caller's process.

    Seeding the global RandomState is the only hook statsmodels offers, so the previous state is
    restored afterwards — otherwise importing this project would silently make every unrelated
    ``np.random`` call in a notebook deterministic.
    """
    np.random.seed(4321)
    expected = np.random.get_state()

    fit_markov(np.asarray(two_regime_returns_pct)[:300])
    actual = np.random.get_state()

    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


# ------------------------------------------------------------------ B-7 test_stationary


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """The stationary distribution of a LEFT-stochastic chain: the pi solving ``P @ pi == pi``.

    Under this repo's column convention that vector is the eigenvector of ``P`` for eigenvalue 1.
    (It is the same object textbooks call the LEFT eigenvector, because they write the transition
    matrix transposed. Naming it carefully here is not pedantry — picking the wrong one is exactly
    the mistake this test exists to catch.)
    """
    values, vectors = np.linalg.eig(P)
    unit = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, unit])
    return vector / vector.sum()


def test_stationary_distribution_matches_the_closed_form(sorted_markov):
    """Pin the test's own arithmetic before trusting it to judge the sampler.

    For a two-state chain, pi is ``[P[0,1], P[1,0]] / (P[0,1] + P[1,0])``: the chance of arriving
    in a state, normalized. If the eigenvector solution and this disagree, the test is broken
    rather than the code.
    """
    P = sorted_markov.P
    inflow = np.array([P[0, 1], P[1, 0]])

    assert stationary_distribution(P) == pytest.approx(inflow / inflow.sum())


def test_stationary(sorted_markov):
    """Simulate with the PRODUCTION sampler and land within 1% of pi (spec B-7).

    Run as 500 independent chains of 1,400 steps rather than one chain of 700,000: the sampler is
    vectorized across paths, so this exercises it in exactly the shape ``run_forecast`` uses, and
    independent chains carry far more information per step than one autocorrelated chain. The
    first 200 steps are burn-in — every chain starts in regime 0, which is the least favourable
    start and therefore the honest one.

    This is the test that catches a transposed sampler. It is also the reason
    :func:`~src.models.monte_carlo.step_regimes` is a public function instead of three lines
    inlined in a loop.
    """
    from src.models.monte_carlo import step_regimes

    P = sorted_markov.P
    expected = stationary_distribution(P)

    chains, steps, burn_in = 500, 1_400, 200
    rng = np.random.default_rng(2026)
    states = np.zeros(chains, dtype=int)
    occupancy = np.zeros(K_REGIMES, dtype=float)

    for step in range(steps):
        states = step_regimes(P, states, rng)
        if step >= burn_in:
            occupancy += np.bincount(states, minlength=K_REGIMES)

    assert occupancy.sum() >= 200_000
    assert occupancy / occupancy.sum() == pytest.approx(expected, abs=0.01)


# ------------------------------------------------------------------ B-5 walk-forward backtest


def summary_stub(*, prob_positive=0.5, return_p10=-0.05, return_p50=0.0, return_p90=0.05):
    """The four fields :func:`score_forecast` reads off a ``ForecastSummary``.

    A stub rather than a real simulation because the scoring rules are arithmetic on four numbers,
    and those numbers have to be chosen — a forecast fitted on synthetic data cannot be made to
    land exactly on an interval edge, which is where a coverage rule is most likely to be wrong.
    """
    return SimpleNamespace(
        prob_positive=prob_positive,
        return_p10=return_p10,
        return_p50=return_p50,
        return_p90=return_p90,
    )


def metric_row(
    model: str,
    *,
    model_used: str | None = None,
    ticker: str = "NVDA",
    origin: date = date(2026, 1, 2),
    brier: float = 0.2,
    mae: float = 0.01,
    covered_80: bool = True,
) -> BacktestRow:
    """One hand-built ``gold.backtest_metrics`` row, for the pooling arithmetic."""
    return BacktestRow(
        origin_date=origin,
        ticker=ticker,
        model=model,
        brier=brier,
        mae=mae,
        covered_80=covered_80,
        model_used=model_used or model,
        converged=True,
        failure_reason=None,
        realized_return=0.01,
        return_p50=0.0,
        prob_positive=0.5,
    )


def test_the_arms_and_their_ladders_are_the_vocabulary_gold_stores():
    """``model`` and ``model_used`` are published strings, so pin them here, not in a docstring."""
    assert MODEL_ARMS == ("news_markov", "markov", "gbm")
    assert LADDERS["news_markov"] == ("news_markov", "markov", "gbm")
    assert LADDERS["markov"] == ("markov", "gbm")
    assert LADDERS["gbm"] == ("gbm",)


def test_fit_arm_rejects_an_unknown_arm(single_window, backtest_cfg):
    with pytest.raises(ValueError, match="unknown model arm"):
        fit_arm("lstm", single_window, backtest_cfg)


# ----------------------------------------------------------------------- origins


def test_weekly_origins_take_the_last_session_of_each_week(backtest_frame):
    """One origin per ISO week, and it is the week's last eligible session.

    Stepping by five rows instead would drift off week ends the moment a holiday shortens a week,
    and two tickers with different histories would then be scored on different days.
    """
    origins = weekly_origins(backtest_frame, n_weeks=4, min_train_days=60, horizon_days=5)

    assert len(origins) == 4
    assert len({origin.isocalendar()[:2] for origin in origins}) == 4
    # The frame is built from business days, so every full week ends on a Friday. The final week
    # is the exception: it is cut short by the five sessions the outcome needs.
    assert [origin.weekday() for origin in origins[:-1]] == [4, 4, 4]


def test_weekly_origins_require_the_training_minimum(backtest_frame):
    """An origin without ``min_train_days`` usable returns behind it is not an origin.

    Usable means non-null ``log_return``: the first row of a ticker carries a NULL because there is
    no previous close (A-4), and counting it would let a window one row short through.
    """
    dates = feature_dates(backtest_frame)
    usable = (~backtest_frame["log_return"].isna()).cumsum().tolist()

    origins = weekly_origins(backtest_frame, n_weeks=99, min_train_days=90, horizon_days=5)

    assert origins
    assert all(usable[dates.index(origin)] >= 90 for origin in origins)
    assert len(origins) < len(
        weekly_origins(backtest_frame, n_weeks=99, min_train_days=60, horizon_days=5)
    )


def test_weekly_origins_never_offer_an_origin_that_cannot_be_scored(backtest_frame):
    """The last five sessions can never be origins: their outcome has not happened yet.

    Including them would score a forecast against a truncated horizon and quietly inflate ``n``.
    """
    dates = feature_dates(backtest_frame)

    origins = weekly_origins(backtest_frame, n_weeks=99, min_train_days=60, horizon_days=5)

    assert set(origins).isdisjoint(dates[-5:])
    assert dates.index(origins[-1]) + 5 == len(dates) - 1


def test_weekly_origins_returns_the_last_n_weeks_in_order(backtest_frame):
    few = weekly_origins(backtest_frame, n_weeks=2, min_train_days=60, horizon_days=5)
    many = weekly_origins(backtest_frame, n_weeks=99, min_train_days=60, horizon_days=5)

    assert len(few) == 2
    assert few == many[-2:]  # the LAST n_weeks, and in chronological order
    assert many == sorted(many)


def test_weekly_origins_rejects_a_non_positive_window(backtest_frame):
    with pytest.raises(ValueError, match="n_weeks must be positive"):
        weekly_origins(backtest_frame, n_weeks=0, min_train_days=60, horizon_days=5)


def test_weekly_origins_is_empty_when_the_history_is_too_short(backtest_frame):
    """No origins is a reportable outcome, not a crash: some tickers are newly listed."""
    assert weekly_origins(backtest_frame, n_weeks=4, min_train_days=10_000, horizon_days=5) == []


def test_feature_dates_refuses_an_unsorted_frame(backtest_frame):
    """Training windows are sliced positionally, which is only safe on a sorted frame."""
    shuffled = backtest_frame.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="sorted by trade_date"):
        feature_dates(shuffled)


def test_feature_dates_refuses_a_frame_missing_a_column(backtest_frame):
    with pytest.raises(ValueError, match="news_sentiment_3d"):
        feature_dates(backtest_frame.drop(columns=["news_sentiment_3d"]))


# ----------------------------------------------------------------------- the window


def test_origin_window_reads_the_price_and_news_at_t(backtest_frame):
    dates = feature_dates(backtest_frame)
    origin = dates[100]

    window = origin_window(backtest_frame, origin, dates=dates, horizon_days=5)

    assert window.origin == origin
    assert window.current_price == pytest.approx(backtest_frame["close"].iloc[100])
    assert window.current_news == pytest.approx(backtest_frame["news_sentiment_3d"].iloc[100])


def test_origin_window_realized_return_is_the_horizon_return(backtest_frame):
    """Scored against the realized 5-day return, computed from closes only (spec B-5)."""
    dates = feature_dates(backtest_frame)
    closes = backtest_frame["close"].tolist()

    window = origin_window(backtest_frame, dates[100], dates=dates, horizon_days=5)

    assert window.realized_return == pytest.approx(closes[105] / closes[100] - 1.0)


def test_origin_window_aligns_the_news_column_with_the_returns(backtest_frame):
    """Model C's news must be trimmed by exactly the warm-up rows the returns lost.

    One row of slippage here would shift the whole news series against the returns, which is the
    lookahead the one-day lag exists to prevent — and it would still fit and still produce numbers.
    """
    dates = feature_dates(backtest_frame)

    window = origin_window(backtest_frame, dates[100], dates=dates, horizon_days=5)

    assert window.news.size == window.returns_pct.size == 100
    assert window.news == pytest.approx(
        backtest_frame["news_sentiment_3d"].to_numpy()[1:101]
    )
    assert window.news[-1] == pytest.approx(window.current_news)


def test_origin_window_refuses_an_origin_without_an_outcome(backtest_frame):
    dates = feature_dates(backtest_frame)

    with pytest.raises(ValueError, match="cannot be scored"):
        origin_window(backtest_frame, dates[-2], dates=dates, horizon_days=5)


def test_origin_window_refuses_a_date_that_is_not_a_session(backtest_frame):
    with pytest.raises(ValueError, match="not a session"):
        origin_window(backtest_frame, date(1999, 1, 4), horizon_days=5)


# ----------------------------------------------------------------------- the ladder


def test_all_three_arms_fit_the_identical_training_window(
    monkeypatch, single_window, backtest_cfg
):
    """The B-5 parity rule: one window per origin, shared by every arm.

    Asserted on the arrays the fit functions actually received, because parity is only meaningful
    at the point of estimation — three call sites that each rebuild "the same" window is precisely
    how they stop being the same.
    """
    received: dict[str, np.ndarray] = {}
    real_news_markov = backtest.fit_news_markov
    real_markov = backtest.fit_markov
    real_gbm = backtest.fit_gbm

    def news_spy(returns_pct, news_series, **kwargs):
        received["news_markov"] = np.array(returns_pct, copy=True)
        received["news"] = np.array(news_series, copy=True)
        return real_news_markov(returns_pct, news_series, **kwargs)

    def markov_spy(returns_pct, **kwargs):
        received["markov"] = np.array(returns_pct, copy=True)
        return real_markov(returns_pct, **kwargs)

    def gbm_spy(returns_pct, **kwargs):
        received["gbm"] = np.array(returns_pct, copy=True)
        return real_gbm(returns_pct, **kwargs)

    monkeypatch.setattr(backtest, "fit_news_markov", news_spy)
    monkeypatch.setattr(backtest, "fit_markov", markov_spy)
    monkeypatch.setattr(backtest, "fit_gbm", gbm_spy)

    for arm in MODEL_ARMS:
        assert fit_arm(arm, single_window, backtest_cfg, min_obs=60) is not None

    assert np.array_equal(received["news_markov"], received["markov"])
    assert np.array_equal(received["markov"], received["gbm"])
    # Model C's exog is built from the same rows; its endog is one shorter only because the
    # mandatory lag consumes the first row, and that happens inside the model.
    assert received["news"].size == received["news_markov"].size


def test_fallback(monkeypatch, single_window, backtest_cfg):
    """Inject a ``FitError`` from Model C and assert Model B was used AND recorded (spec B-7).

    A silent fallback is as bad as a crash: a published Model C verdict that was actually produced
    by Model B is a wrong claim, so both the substitution and its reason are asserted.
    """

    def boom(*args, **kwargs):
        raise FitError("news_markov: injected optimizer failure")

    monkeypatch.setattr(backtest, "fit_news_markov", boom)

    fit = fit_arm("news_markov", single_window, backtest_cfg, min_obs=60)

    assert fit is not None
    assert fit.model == "news_markov"
    assert fit.model_used == "markov"
    assert fit.summary.model_used == "markov"
    assert fit.fell_back
    assert fit.failure_reason.startswith("news_markov: news_markov: injected")


def test_the_ladder_descends_all_the_way_to_gbm(monkeypatch, single_window, backtest_cfg):
    """C then B then A, recording every rung that failed on the way down."""

    def boom(*args, **kwargs):
        raise FitError("no")

    monkeypatch.setattr(backtest, "fit_news_markov", boom)
    monkeypatch.setattr(backtest, "fit_markov", boom)

    fit = fit_arm("news_markov", single_window, backtest_cfg, min_obs=60)

    assert fit.model_used == "gbm"
    assert fit.failure_reason.count("no") == 2
    assert fit.converged is None  # GBM has no optimizer, so there is nothing to converge


def test_fit_arm_reports_no_forecast_when_every_rung_fails(
    monkeypatch, single_window, backtest_cfg
):
    """One unfittable window must not abort a 26-week backtest."""

    def boom(*args, **kwargs):
        raise FitError("no")

    for name in ("fit_news_markov", "fit_markov", "fit_gbm"):
        monkeypatch.setattr(backtest, name, boom)

    assert fit_arm("news_markov", single_window, backtest_cfg, min_obs=60) is None


def test_a_skipped_origin_is_carried_not_silently_dropped(
    monkeypatch, backtest_frame, backtest_cfg
):
    """A shorter ``n`` has to be visible: "no better" and "never evaluated" are different claims."""

    def boom(*args, **kwargs):
        raise FitError("no")

    for name in ("fit_news_markov", "fit_markov", "fit_gbm"):
        monkeypatch.setattr(backtest, name, boom)

    rows, skipped = backtest.backtest_ticker("NVDA", backtest_frame, backtest_cfg)

    assert rows == []
    assert len(skipped) == 2 * len(MODEL_ARMS)  # n_weeks=2 origins, every arm skipped
    assert {entry["model"] for entry in skipped} == set(MODEL_ARMS)


def test_a_markov_arm_records_the_optimizer_verdict(single_window, backtest_cfg):
    """``converged`` is reported, never raised on — but it has to reach the row."""
    fit = fit_arm("markov", single_window, backtest_cfg, min_obs=60)

    assert fit.model_used == "markov"
    assert isinstance(fit.converged, bool)
    assert fit.failure_reason is None


# ----------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    ("prob_positive", "realized", "expected"),
    [
        (0.7, 0.02, 0.09),  # right side, 0.3 off
        (0.7, -0.02, 0.49),  # wrong side, 0.7 off
        (0.5, 0.02, 0.25),  # an honest "no idea" always scores 0.25
        (1.0, 0.02, 0.0),  # certain and correct
        (1.0, -0.02, 1.0),  # certain and wrong: the worst possible score
    ],
)
def test_brier_is_the_squared_error_of_the_directional_probability(
    prob_positive, realized, expected
):
    """Scored as a probability, not as a directional call: overconfidence must cost something."""
    scores = score_forecast(summary_stub(prob_positive=prob_positive), realized)

    assert scores["brier"] == pytest.approx(expected)


def test_a_flat_realized_return_does_not_count_as_positive():
    """The forecast probability is P(R5 > 0), so the outcome must be scored on the same strict >."""
    assert score_forecast(summary_stub(prob_positive=1.0), 0.0)["brier"] == pytest.approx(1.0)


def test_mae_is_the_absolute_error_of_the_median_return():
    scores = score_forecast(summary_stub(return_p50=0.01), -0.004)

    assert scores["mae"] == pytest.approx(0.014)


@pytest.mark.parametrize(
    ("realized", "covered"),
    [
        (0.0, True),
        (-0.05, True),  # the interval is inclusive at both ends
        (0.05, True),
        (-0.0500001, False),
        (0.0500001, False),
    ],
)
def test_covered_80_is_the_closed_interval(realized, covered):
    """80% coverage is the property being measured, so the boundary is part of the contract."""
    scores = score_forecast(summary_stub(return_p10=-0.05, return_p90=0.05), realized)

    assert scores["covered_80"] is covered


def test_score_forecast_refuses_a_non_finite_outcome():
    with pytest.raises(ValueError, match="must be finite"):
        score_forecast(summary_stub(), float("nan"))


# ----------------------------------------------------------------------- pooling


def test_pooled_summary_reports_n_and_the_tickers_behind_it():
    """n is a first-class output: a Brier difference over 4 forecasts is not a finding."""
    rows = [
        metric_row("markov", ticker="NVDA", brier=0.2, mae=0.01, covered_80=True),
        metric_row("markov", ticker="MSFT", brier=0.4, mae=0.03, covered_80=False),
    ]

    (pooled,) = pooled_summary(rows)

    assert pooled.model == "markov"
    assert pooled.n == 2
    assert pooled.n_tickers == 2
    assert pooled.brier == pytest.approx(0.3)
    assert pooled.mae == pytest.approx(0.02)
    assert pooled.coverage_80 == pytest.approx(0.5)


def test_pooled_fallback_rate_counts_the_arms_that_descended():
    """How often C failed to converge, which is the number the page shows next to C's score."""
    rows = [
        metric_row("news_markov"),
        metric_row("news_markov", model_used="markov", origin=date(2026, 1, 9)),
        metric_row("news_markov", model_used="gbm", origin=date(2026, 1, 16)),
        metric_row("news_markov", origin=date(2026, 1, 23)),
        metric_row("gbm"),
    ]

    pooled = {row.model: row for row in pooled_summary(rows)}

    assert pooled["news_markov"].fallback_rate == pytest.approx(0.5)
    # GBM is the bottom of the ladder, so 0.0 here is a fact rather than a missing value.
    assert pooled["gbm"].fallback_rate == pytest.approx(0.0)


def test_pooled_summary_omits_an_arm_that_was_never_scored():
    """An unscored arm has no Brier score, and printing 0.0 would read as a perfect one."""
    pooled = pooled_summary([metric_row("gbm")])

    assert [row.model for row in pooled] == ["gbm"]


def test_pooled_summary_is_ordered_richest_model_first():
    rows = [metric_row(model) for model in reversed(MODEL_ARMS)]

    assert [row.model for row in pooled_summary(rows)] == list(MODEL_ARMS)


# ----------------------------------------------------------------------- the whole run


def test_run_backtest_pools_across_tickers(backtest_frame):
    """Pooled across tickers (spec B-5), with one row per (origin, ticker, model)."""
    cfg = backtest_config(min_train_days=60, n_weeks=1, n_paths=100)
    frames = {"NVDA": backtest_frame, "MSFT": backtest_frame}

    result = run_backtest(frames, cfg)

    grain = [(row.origin_date, row.ticker, row.model) for row in result.rows]

    assert len(result.rows) == 2 * len(MODEL_ARMS)  # 1 origin, 2 tickers, 3 arms
    assert {row.ticker for row in result.rows} == {"NVDA", "MSFT"}
    assert len(set(grain)) == len(grain)  # the MERGE key of gold.backtest_metrics is unique
    assert all(pooled.n == 2 and pooled.n_tickers == 2 for pooled in result.pooled)
    assert result.skipped == ()


# ------------------------------------------------------------------ frozen-rule guards


def test_models_package_never_imports_pyspark():
    """Hard rule: ``src/models/`` is pure pandas/numpy/statsmodels (spec rule 4, C-b)."""
    offenders = {
        path.name
        for path in sorted(MODELS_DIR.glob("*.py"))
        if re.search(r"^\s*(import|from)\s+pyspark", path.read_text(encoding="utf-8"), re.M)
    }

    assert offenders == set()
