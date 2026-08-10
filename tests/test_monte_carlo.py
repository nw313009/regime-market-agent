"""Monte Carlo tests (spec B-7).

- ``test_mc_seed``: the same seed must produce identical percentiles; a different seed must
  produce different ones. Both halves matter — the second catches a Generator that is not
  actually being used.
- Assert the sampler reads ``P_h[:, current_regime]``, i.e. the previous state selects a
  COLUMN. Pair this with ``test_stationary`` in ``test_models.py``.
- Assert the news decay follows ``news_h = current_news * exp(-ln(2)/half_life * h)``.
- Assert the output carries ``n_paths``, ``seed`` and ``model_version``, and that raw paths
  are NOT persisted.
- Assert returns are simulated on the decimal scale: fitted percent-scale mu and sigma must
  have been divided by 100 before simulation.

THE COLUMN READ IS TESTED STATISTICALLY, not structurally. On two regimes a transposed matrix is
either not a set of distributions at all (its columns no longer sum to 1) or it is symmetric and
identical to the original, so there is no clever assertion that separates the two by inspection.
``test_next_state_distribution_is_the_column`` instead drives the real sampler from each previous
state in turn and checks the empirical next-state frequencies against ``P[:, j]``, which a
row-reading implementation cannot reproduce.

THE CONFIG IS THE REAL ONE. These tests read ``config/config.yaml`` rather than a hand-written
dict, so the 5,000 paths, the 5-day horizon, the seed 42 and the 2-day half-life that ship with
the project are the numbers under test. A forecast test that invents its own config proves the
simulation works on a configuration nobody runs.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models import to_decimal
from src.models.gbm import MODEL_NAME as GBM_MODEL
from src.models.gbm import fit_gbm
from src.models.markov import MODEL_NAME as MARKOV_MODEL
from src.models.markov import K_REGIMES, SortedParams
from src.models.monte_carlo import (
    LOSS_THRESHOLD,
    MODEL_VERSION,
    PERCENTILES,
    ForecastSummary,
    decayed_news,
    draw_states,
    forecast_config,
    rng_for,
    run_forecast,
    run_gbm_forecast,
    step_regimes,
    tvtp_transition_matrix,
)
from src.models.news_markov import MODEL_NAME as NEWS_MARKOV_MODEL

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    """The project's real config.yaml (spec B0 keys)."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def reseeded(cfg: dict, seed: int) -> dict:
    """The same config with a different forecast seed."""
    return {**cfg, "forecast": {**cfg["forecast"], "seed": seed}}


# ------------------------------------------------------------------ config plumbing


def test_forecast_config_reads_the_shipped_values(cfg):
    settings = forecast_config(cfg)

    assert (settings.horizon_days, settings.n_paths, settings.seed) == (5, 5000, 42)
    assert settings.half_life_days == 2.0


def test_rng_for_is_seeded_from_config(cfg):
    """One Generator per run, seeded from config, so a stored seed reproduces a stored forecast."""
    expected = np.random.default_rng(forecast_config(cfg).seed).random(5)

    assert rng_for(cfg).random(5) == pytest.approx(expected)


@pytest.mark.parametrize(
    "override",
    [{"horizon_days": 0}, {"n_paths": 0}],
)
def test_forecast_config_rejects_non_positive_settings(cfg, override):
    with pytest.raises(ValueError, match="must be positive"):
        forecast_config({**cfg, "forecast": {**cfg["forecast"], **override}})


# ------------------------------------------------------------------ news decay


def test_news_decay_halves_every_half_life(cfg):
    """``news_h`` at h = half_life is exactly half of its h = 0 value (spec B-4)."""
    half_life = forecast_config(cfg).half_life_days
    current_news = 0.8

    assert decayed_news(current_news, 0, half_life) == pytest.approx(current_news)
    assert decayed_news(current_news, half_life, half_life) == pytest.approx(current_news / 2)
    assert decayed_news(current_news, 2 * half_life, half_life) == pytest.approx(current_news / 4)


def test_news_decay_matches_the_formula_exactly():
    """``current_news * exp(-ln(2)/half_life * h)``, written out and compared term by term."""
    current_news, half_life = -0.35, 3.0
    horizon = np.arange(1, 6)

    expected = [current_news * math.exp(-math.log(2.0) / half_life * h) for h in horizon]
    assert decayed_news(current_news, horizon, half_life) == pytest.approx(expected)


def test_news_decay_preserves_sign_and_zero():
    """Decay shrinks a signal toward zero; it never flips it, and zero news stays zero."""
    assert decayed_news(-1.0, 5, 2.0) < 0
    assert decayed_news(0.0, 5, 2.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ the regime sampler


def test_next_state_distribution_is_the_column():
    """``next ~ Categorical(P[:, previous])`` — the previous state selects a COLUMN (§5)."""
    P = np.array([[0.90, 0.30], [0.10, 0.70]])
    rng = np.random.default_rng(1)
    paths = 50_000

    for previous in range(K_REGIMES):
        states = step_regimes(P, np.full(paths, previous), rng)
        empirical = np.bincount(states, minlength=K_REGIMES) / paths

        assert empirical == pytest.approx(P[:, previous], abs=0.01)


def test_sampler_respects_a_deterministic_transition():
    """A column of ``[0, 1]`` must send every path to regime 1, whatever it came from."""
    always_high = np.array([[0.0, 0.0], [1.0, 1.0]])
    states = step_regimes(always_high, np.array([0, 1, 0, 1]), np.random.default_rng(2))

    assert states.tolist() == [1, 1, 1, 1]


def test_sampler_rejects_a_right_stochastic_matrix():
    """Rows summing to 1 instead of columns is the transposition this repo must never accept."""
    with pytest.raises(ValueError, match="LEFT-stochastic"):
        step_regimes(
            np.array([[0.90, 0.10], [0.30, 0.70]]), np.zeros(3, dtype=int), np.random.default_rng(3)
        )


def test_draw_states_reproduces_its_input_distribution():
    """The initial-regime draw: ``Categorical(filtered_current)`` over the whole path population."""
    filtered = np.array([0.3, 0.7])
    paths = 100_000
    probs = np.broadcast_to(filtered[:, None], (K_REGIMES, paths))

    states = draw_states(probs, np.random.default_rng(4))
    empirical = np.bincount(states, minlength=K_REGIMES) / paths

    assert empirical == pytest.approx(filtered, abs=0.005)


# ------------------------------------------------------------------ test_mc_seed


def test_mc_seed_same_seed_is_identical(sorted_markov, cfg):
    first = run_forecast(sorted_markov, None, 100.0, 0.2, cfg, rng_for(cfg))
    second = run_forecast(sorted_markov, None, 100.0, 0.2, cfg, rng_for(cfg))

    assert first == second


def test_mc_seed_different_seed_differs(sorted_markov, cfg):
    """The other half of the test: a Generator that is never used also passes the first half."""
    other = reseeded(cfg, 43)

    first = run_forecast(sorted_markov, None, 100.0, 0.2, cfg, rng_for(cfg))
    second = run_forecast(sorted_markov, None, 100.0, 0.2, other, rng_for(other))

    assert first != second
    for name in ("price_p10", "price_p50", "price_p90", "prob_positive"):
        assert getattr(first, name) != getattr(second, name)
    assert second.seed == 43


# ------------------------------------------------------------------ the stored row


def test_summary_carries_the_provenance_columns(sorted_markov, cfg):
    summary = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))

    assert summary.n_paths == 5000
    assert summary.seed == 42
    assert summary.horizon_days == 5
    assert summary.model_version == MODEL_VERSION
    assert summary.model_used == MARKOV_MODEL


def test_summary_persists_no_raw_paths(sorted_markov, cfg):
    """Spec B-4: do NOT persist the 5,000 paths. Every field is a scalar, so it cannot."""
    summary = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))

    for field in fields(summary):
        value = getattr(summary, field.name)
        assert np.isscalar(value), f"{field.name} is not a scalar: {type(value)}"


def test_summary_is_internally_consistent(sorted_markov, cfg):
    """Quantiles ordered, prices and returns describing the same distribution, probabilities in [0, 1]."""
    price = 187.5
    summary = run_forecast(sorted_markov, None, price, 0.0, cfg, rng_for(cfg))

    assert summary.price_p10 < summary.price_p50 < summary.price_p90
    assert summary.return_p10 < summary.return_p50 < summary.return_p90
    assert summary.current_price == pytest.approx(price)

    for quantile in PERCENTILES:
        price_value = getattr(summary, f"price_p{quantile}")
        return_value = getattr(summary, f"return_p{quantile}")
        assert return_value == pytest.approx(price_value / price - 1.0)

    for name in ("prob_positive", "prob_loss_gt_5pct", "prob_low_vol", "prob_high_vol"):
        assert 0.0 <= getattr(summary, name) <= 1.0

    assert summary.prob_low_vol + summary.prob_high_vol == pytest.approx(1.0)


def test_summary_regime_probabilities_come_from_the_filtered_state(sorted_markov, cfg):
    """``gold.forecast_runs`` and ``gold.regime_states`` must not disagree about the regime."""
    summary = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))

    assert summary.prob_low_vol == pytest.approx(sorted_markov.prob_low_vol)
    assert summary.prob_high_vol == pytest.approx(sorted_markov.prob_high_vol)


# ------------------------------------------------------------------ the decimal scale


def test_simulation_uses_the_decimal_scale(sorted_markov, cfg):
    """A five-day spread of a few percent, not a few hundred percent (spec B-0).

    If the percent-scale parameters reached the simulation, a 1.0-3.0 sigma would become a 100-300%
    daily move and the 80% interval would span many multiples of the current price. The band below
    is deliberately wide: it is checking the ORDER OF MAGNITUDE, which is what a missing
    divide-by-100 destroys.
    """
    summary = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))
    spread = summary.return_p90 - summary.return_p10

    assert 0.01 < spread < 0.30


def test_run_forecast_refuses_percent_scale_parameters(sorted_markov, cfg):
    """Hand the simulator the ESTIMATION-scale parameters and it must refuse, not simulate."""
    percent_scale = replace(
        sorted_markov, mus=sorted_markov.mus_pct, sigmas=sorted_markov.sigmas_pct
    )

    with pytest.raises(ValueError, match="not divided by 100"):
        run_forecast(percent_scale, None, 100.0, 0.0, cfg, rng_for(cfg))


# ------------------------------------------------------------------ Model B vs Model C branch


def test_model_b_ignores_news_entirely(sorted_markov, cfg):
    """Model B has no news channel, so the news argument must not reach its output."""
    quiet = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))
    loud = run_forecast(sorted_markov, None, 100.0, -0.9, cfg, rng_for(cfg))

    assert quiet == loud


def test_model_c_transitions_respond_to_news(sorted_news_markov, fitted_news_markov, cfg):
    """Model C's whole claim is that news moves the transition probabilities."""
    quiet = run_forecast(sorted_news_markov, fitted_news_markov, 100.0, 0.0, cfg, rng_for(cfg))
    loud = run_forecast(sorted_news_markov, fitted_news_markov, 100.0, -0.9, cfg, rng_for(cfg))

    assert quiet != loud
    assert quiet.model_used == NEWS_MARKOV_MODEL
    assert loud.model_used == NEWS_MARKOV_MODEL


def test_tvtp_transition_matrix_is_sorted_and_left_stochastic(sorted_news_markov, fitted_news_markov):
    """Every rebuilt matrix gets the SAME perm, or the regimes silently swap mid-simulation."""
    matrix = tvtp_transition_matrix(fitted_news_markov, -0.4, sorted_news_markov.perm)
    raw = fitted_news_markov.model.regime_transition_matrix(
        fitted_news_markov.params, exog_tvtp=np.array([[1.0, -0.4]])
    )[:, :, 0]
    perm = sorted_news_markov.perm

    assert np.allclose(matrix.sum(axis=0), 1)
    for i in range(K_REGIMES):
        for j in range(K_REGIMES):
            assert matrix[i, j] == pytest.approx(raw[perm[i], perm[j]])


def test_run_forecast_rejects_a_time_invariant_result(sorted_markov, fitted_markov, cfg):
    """Passing Model B's result where Model C's belongs would silently mislabel the row."""
    with pytest.raises(ValueError, match="not a TVTP fit"):
        run_forecast(sorted_markov, fitted_markov, 100.0, 0.0, cfg, rng_for(cfg))


# ------------------------------------------------------------------ Model A


def test_gbm_forecast_matches_the_analytic_lognormal(cfg):
    """No regimes, so the horizon distribution has a closed form to check against.

    Five daily draws of ``N(mu, sigma)`` sum to ``N(5*mu, sqrt(5)*sigma)`` in log space, so the
    median price is ``P0 * exp(5*mu)`` and the 90th percentile is ``P0 * exp(5*mu + 1.2816*sqrt(5)*sigma)``.
    """
    settings = forecast_config(cfg)
    mu_pct, sigma_pct = 0.02, 1.20
    summary = run_gbm_forecast({"mu": mu_pct, "sigma": sigma_pct}, 100.0, cfg, rng_for(cfg))

    mu, sigma = to_decimal(mu_pct), to_decimal(sigma_pct)
    horizon_mu = settings.horizon_days * mu
    horizon_sigma = math.sqrt(settings.horizon_days) * sigma

    assert summary.model_used == GBM_MODEL
    assert summary.price_p50 == pytest.approx(100.0 * math.exp(horizon_mu), rel=0.01)
    assert summary.price_p90 == pytest.approx(
        100.0 * math.exp(horizon_mu + 1.2816 * horizon_sigma), rel=0.02
    )
    assert summary.prob_positive == pytest.approx(0.5, abs=0.03)


def test_gbm_forecast_has_no_regime_probabilities(cfg):
    """A model without regimes has no honest regime probability, so the column is NULL.

    ``None`` and not NaN: the summary is compared with ``==`` by the reproducibility and
    no-lookahead tests, and a NaN field would make a Model A summary unequal to itself.
    """
    summary = run_gbm_forecast({"mu": 0.0, "sigma": 1.0}, 100.0, cfg, rng_for(cfg))

    assert summary.prob_low_vol is None
    assert summary.prob_high_vol is None
    assert summary == run_gbm_forecast({"mu": 0.0, "sigma": 1.0}, 100.0, cfg, rng_for(cfg))


def test_gbm_forecast_rescales_the_percent_parameters(cfg):
    """``fit_gbm`` returns percent; the simulation must divide by 100 here (spec B-0)."""
    tiny = run_gbm_forecast({"mu": 0.0, "sigma": 1.0}, 100.0, cfg, rng_for(cfg))

    # sigma 1.0 PERCENT is a 1% daily move: a five-day 80% interval of a few percent.
    assert 0.01 < tiny.return_p90 - tiny.return_p10 < 0.15


def test_gbm_end_to_end_from_a_fitted_window(two_regime_returns_pct, cfg):
    """fit -> forecast on the same window the Markov models use (the B-1 parity rule)."""
    summary = run_gbm_forecast(fit_gbm(two_regime_returns_pct), 100.0, cfg, rng_for(cfg))

    assert isinstance(summary, ForecastSummary)
    assert summary.model_used == GBM_MODEL
    assert summary.price_p10 < summary.price_p50 < summary.price_p90


def test_prob_loss_threshold_is_five_percent(sorted_markov, cfg):
    """``prob_loss_gt_5pct`` must be P(R5 < -5%), and consistent with the reported quantiles."""
    assert LOSS_THRESHOLD == 0.05

    summary = run_forecast(sorted_markov, None, 100.0, 0.0, cfg, rng_for(cfg))

    # If a 5% loss is worse than the 10th percentile outcome, fewer than 10% of paths hit it.
    if summary.return_p10 > -LOSS_THRESHOLD:
        assert summary.prob_loss_gt_5pct < 0.10
    else:
        assert summary.prob_loss_gt_5pct >= 0.10


def test_run_forecast_rejects_an_impossible_price(sorted_markov, cfg):
    with pytest.raises(ValueError, match="current_price"):
        run_forecast(sorted_markov, None, 0.0, 0.0, cfg, rng_for(cfg))


def test_sorted_params_is_the_documented_input(sorted_markov):
    """Guard the fixture: these tests are only meaningful against the real B-2 dataclass."""
    assert isinstance(sorted_markov, SortedParams)
