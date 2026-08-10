"""Monte Carlo forecast simulation (spec B-4).

Contract::

    def run_forecast(sorted_params, model_res_or_none, current_price,
                     current_news, cfg, rng) -> ForecastSummary

5,000 paths x 5 trading days. Per path, for each day h in 1..5::

    news_h = current_news * exp(-ln(2) / half_life * h)

    if TVTP model:
        P_h = res.model.regime_transition_matrix(
            res.params, exog_tvtp=np.array([[1.0, news_h]]))[:, :, 0]
        # then apply the SAME perm that sort_regimes used
    else:
        P_h = the sorted static P

    next_regime ~ Categorical(P_h[:, current_regime])   # COLUMN = previous state
    r ~ Normal(mus[next_regime], sigmas[next_regime])   # decimal scale, not percent
    price *= exp(r)
    current_regime = next_regime

The column indexing is the whole ballgame: the matrix is left-stochastic, so the previous
state selects a COLUMN. A sampler that indexes rows is transposed, and the stationary test
in ``tests/test_models.py`` catches it immediately.

Initial regime per path ~ ``Categorical(filtered_current)``.

GBM paths: ``r ~ Normal(mu, sigma)``, no regimes.

Randomness: ``rng = np.random.default_rng(cfg.forecast.seed)`` — one Generator per forecast
run, and the seed is stored with the output so the run is reproducible.

Outputs: price and return P10/P50/P90; ``prob_positive`` = ``mean(R5 > 0)``;
``prob_loss_gt_5pct``; regime probabilities; ``n_paths``; ``model_version``. Emphasize the
distribution, not a point prediction.

Do NOT persist the raw 5,000 paths.

News decay is an assumption, not a measurement, so the UI discloses it in a sentence.

HOW THIS MODULE IS STRUCTURED, and why it is not one loop. The regime step is
:func:`step_regimes` and the categorical draw is :func:`draw_states`, both public, both
vectorized across paths. The stationary test in ``tests/test_models.py`` has to drive the
PRODUCTION sampler for hundreds of thousands of steps — a sampler that only exists inline inside
``run_forecast`` can be tested for its outputs but never for its orientation, and orientation is
the failure this repo is most exposed to.

DRAW ORDER IS PART OF THE CONTRACT. Given one Generator, the draws are: the initial states for
all paths, then, for each day, the day's states for all paths followed by the day's returns for
all paths. Reproducibility from a stored seed depends on that order, so reordering the draws
changes every stored forecast even though the model is unchanged. That is what ``MODEL_VERSION``
is for.

MODEL A'S RESCALE LIVES HERE. ``fit_gbm`` returns percent-scale parameters, as the estimation
scale requires (spec B-0), and :func:`run_gbm_forecast` divides them by 100 — the same rescale
``sort_regimes`` already performed for Models B and C. Both paths therefore simulate decimals,
and neither caller has to remember the factor.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.models import to_decimal
from src.models.gbm import MODEL_NAME as GBM_MODEL
from src.models.markov import K_REGIMES
from src.models.markov import MODEL_NAME as MARKOV_MODEL
from src.models.news_markov import MODEL_NAME as NEWS_MARKOV_MODEL

__all__ = [
    "LOSS_THRESHOLD",
    "MODEL_VERSION",
    "PERCENTILES",
    "ForecastConfig",
    "ForecastSummary",
    "decayed_news",
    "draw_states",
    "forecast_config",
    "rng_for",
    "run_forecast",
    "run_gbm_forecast",
    "step_regimes",
    "tvtp_transition_matrix",
]

log = logging.getLogger(__name__)

#: Stored on every forecast row (spec B-6). Bump it when the simulation changes in a way that
#: makes previously stored forecasts non-comparable — a new draw order counts, because a stored
#: seed no longer reproduces the stored numbers.
MODEL_VERSION = "2.1.0"

#: The reported quantiles of the horizon distribution. A distribution, not a point prediction.
PERCENTILES = (10, 50, 90)

#: ``prob_loss_gt_5pct`` is P(R5 < -5%), strictly worse than a 5% loss.
LOSS_THRESHOLD = 0.05


@dataclass(frozen=True)
class ForecastConfig:
    """The four config values a forecast run needs (spec B0 keys), validated once."""

    horizon_days: int
    n_paths: int
    seed: int
    half_life_days: float


@dataclass(frozen=True)
class ForecastSummary:
    """One row of ``gold.forecast_runs`` (spec B-6), minus the identifiers the caller owns.

    Every field is a scalar, deliberately: "do NOT persist the raw paths" is easier to keep true
    if the summary has nowhere to put them. ``ticker``, ``forecast_id``, ``generated_at`` and
    ``as_of_date`` belong to the task that writes the row, not to the simulation.

    Equality is the default field-by-field comparison, because two of the B-7 tests compare whole
    summaries: identical seeds must give an identical summary, and corrupting the future must not
    change one.
    """

    model_used: str
    model_version: str
    horizon_days: int
    n_paths: int
    seed: int
    current_price: float
    price_p10: float
    price_p50: float
    price_p90: float
    return_p10: float
    return_p50: float
    return_p90: float
    prob_positive: float
    prob_loss_gt_5pct: float
    #: Current filtered regime probabilities, carried through so ``gold.forecast_runs`` and
    #: ``gold.regime_states`` cannot disagree about the regime the forecast was built from.
    #: ``None`` for Model A, which has no regimes, and stored as a NULL.
    #:
    #: ``None`` rather than NaN specifically because of the equality contract above: NaN is not
    #: equal to itself, so a NaN here would make every Model A summary compare unequal to itself
    #: and quietly defeat the two tests this dataclass exists to serve.
    prob_low_vol: float | None
    prob_high_vol: float | None


def forecast_config(cfg: Mapping) -> ForecastConfig:
    """Read ``forecast.*`` and ``news.half_life_days`` out of config.yaml (spec B0)."""
    forecast = cfg["forecast"]
    settings = ForecastConfig(
        horizon_days=int(forecast["horizon_days"]),
        n_paths=int(forecast["n_paths"]),
        seed=int(forecast["seed"]),
        half_life_days=float(cfg["news"]["half_life_days"]),
    )
    for name in ("horizon_days", "n_paths", "half_life_days"):
        value = getattr(settings, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return settings


def rng_for(cfg: Mapping) -> np.random.Generator:
    """One ``Generator`` per forecast run, seeded from config (spec B-4).

    Built here rather than at each call site so the seed stored on the summary and the seed the
    paths were actually drawn with are the same number by construction.
    """
    return np.random.default_rng(forecast_config(cfg).seed)


def decayed_news(current_news: float, h, half_life_days: float):
    """``news_h = current_news * exp(-ln(2) / half_life * h)`` (spec B-4).

    ``h`` may be a scalar or an array of horizon days. The half-life is an ASSUMPTION about how
    fast a news signal stops mattering, not something this project measured, which is why the UI
    discloses it in a sentence rather than presenting the forecast as unconditional.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    return current_news * np.exp(-np.log(2.0) / half_life_days * np.asarray(h, dtype=float))


def draw_states(probs, rng: np.random.Generator) -> np.ndarray:
    """Categorical draw, one per COLUMN of ``probs`` (shape ``(k, n)``) -> ``(n,)`` states.

    Columns rather than rows because that is the orientation everything else in this repo uses:
    ``P[:, j]`` is the distribution of the next state given previous state ``j``, so a whole
    population of paths becomes ``P[:, states]`` and needs no transpose anywhere.
    """
    columns = np.asarray(probs, dtype=float)
    if columns.ndim != 2 or columns.shape[0] != K_REGIMES:
        raise ValueError(f"expected probabilities shaped ({K_REGIMES}, n), got {columns.shape}")
    if not np.all(np.isfinite(columns)) or np.any(columns < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(columns.sum(axis=0), 1.0):
        raise ValueError("every column must sum to 1")

    cumulative = np.cumsum(columns, axis=0)
    # Pin the top of the ladder to exactly 1: the last cumulative cell can land a float epsilon
    # below 1, and a uniform draw above every threshold would index a regime that does not exist.
    cumulative[-1, :] = 1.0
    draws = rng.random(columns.shape[1])
    return (draws[None, :] > cumulative).sum(axis=0)


def step_regimes(P, current_states, rng: np.random.Generator) -> np.ndarray:
    """One regime transition for every path: ``next ~ Categorical(P[:, current])``.

    THE COLUMN READ IS THE POINT (architecture doc §5). ``P`` is left-stochastic — rows are the
    next state, columns the previous one — so the previous state selects a column. Indexing rows
    instead produces a chain that is plausible, stationary, and wrong.
    """
    matrix = _validated_transition(P)
    states = np.asarray(current_states, dtype=int)
    if states.ndim != 1:
        raise ValueError(f"current_states must be 1-D, got shape {states.shape}")
    if np.any((states < 0) | (states >= K_REGIMES)):
        raise ValueError(f"states must be in [0, {K_REGIMES})")

    return draw_states(matrix[:, states], rng)


def tvtp_transition_matrix(res, news_h: float, perm) -> np.ndarray:
    """Model C's transition matrix for one horizon day, sorted with the SAME perm (spec B-4).

    ``sort_regimes`` permuted the parameters; statsmodels has not, and it never will — it knows
    nothing about which regime this project calls "low volatility". So every matrix rebuilt here
    must be permuted again with the identical ``perm``, or the simulation will draw a calm return
    after a turbulent transition.
    """
    exog_tvtp = np.array([[1.0, float(news_h)]])
    matrix = res.model.regime_transition_matrix(res.params, exog_tvtp=exog_tvtp)[:, :, 0]
    ordering = np.asarray(perm, dtype=int)
    return _validated_transition(matrix[np.ix_(ordering, ordering)])


def run_forecast(
    sorted_params,
    model_res_or_none,
    current_price: float,
    current_news: float,
    cfg: Mapping,
    rng: np.random.Generator,
) -> ForecastSummary:
    """Simulate the horizon distribution for Model B or Model C (spec B-4).

    ``model_res_or_none`` is the fitted statsmodels result for Model C and ``None`` for Model B.
    That argument alone selects the branch: with a result, the transition matrix is rebuilt for
    each horizon day from the decayed news; without one, the sorted static ``P`` is reused every
    day. It also decides ``model_used``, so the recorded model and the mathematics that produced
    the row cannot disagree.

    Model A goes through :func:`run_gbm_forecast`, which takes parameters rather than
    ``SortedParams`` because it has no regimes to sort.
    """
    settings = forecast_config(cfg)
    price = _validated_price(current_price)

    mus = np.asarray(sorted_params.mus, dtype=float)
    sigmas = np.asarray(sorted_params.sigmas, dtype=float)
    _validated_scales(mus, sigmas)
    filtered = np.asarray(sorted_params.filtered_current, dtype=float)
    perm = np.asarray(sorted_params.perm, dtype=int)
    static_P = _validated_transition(sorted_params.P)

    tvtp = model_res_or_none is not None
    if tvtp and not getattr(model_res_or_none.model, "tvtp", False):
        raise ValueError(
            "model_res_or_none is not a TVTP fit — pass None for a time-invariant model so the "
            "static P is used and model_used is recorded as Model B"
        )

    # Initial regime per path ~ Categorical(filtered_current): the current regime is a
    # distribution, not a decision, and collapsing it to its argmax would throw away the
    # uncertainty the whole forecast exists to express.
    states = draw_states(np.broadcast_to(filtered[:, None], (K_REGIMES, settings.n_paths)), rng)
    prices = np.full(settings.n_paths, price, dtype=float)

    for h in range(1, settings.horizon_days + 1):
        if tvtp:
            news_h = decayed_news(current_news, h, settings.half_life_days)
            transition = tvtp_transition_matrix(model_res_or_none, news_h, perm)
        else:
            transition = static_P
        states = step_regimes(transition, states, rng)
        prices *= np.exp(rng.normal(mus[states], sigmas[states]))

    return _summarize(
        terminal_prices=prices,
        current_price=price,
        model_used=NEWS_MARKOV_MODEL if tvtp else MARKOV_MODEL,
        settings=settings,
        prob_low_vol=float(filtered[0]),
        prob_high_vol=float(filtered[1]),
    )


def run_gbm_forecast(
    gbm_params: Mapping[str, float],
    current_price: float,
    cfg: Mapping,
    rng: np.random.Generator,
) -> ForecastSummary:
    """Simulate the horizon distribution for Model A: ``r ~ Normal(mu, sigma)``, no regimes.

    ``gbm_params`` is :func:`~src.models.gbm.fit_gbm`'s output, in the PERCENT estimation scale;
    the divide-by-100 happens here (spec B-0). The regime probabilities on the summary are ``None``,
    because a model without regimes has no honest number to put there.
    """
    settings = forecast_config(cfg)
    price = _validated_price(current_price)

    mu = float(to_decimal(gbm_params["mu"]))
    sigma = float(to_decimal(gbm_params["sigma"]))
    _validated_scales(np.array([mu]), np.array([sigma]))

    draws = rng.normal(mu, sigma, size=(settings.horizon_days, settings.n_paths))
    prices = price * np.exp(draws.sum(axis=0))

    return _summarize(
        terminal_prices=prices,
        current_price=price,
        model_used=GBM_MODEL,
        settings=settings,
        prob_low_vol=None,
        prob_high_vol=None,
    )


# ------------------------------------------------------------------ internals


def _summarize(
    *,
    terminal_prices: np.ndarray,
    current_price: float,
    model_used: str,
    settings: ForecastConfig,
    prob_low_vol: float | None,
    prob_high_vol: float | None,
) -> ForecastSummary:
    """Reduce the paths to the stored row, then drop them.

    The return quantiles are derived from the PRICE quantiles rather than computed separately.
    ``price -> price / P0 - 1`` is affine and increasing, so the two agree exactly, and deriving
    them guarantees the UI can never show a P10 price and a P10 return that disagree by a rounding
    error.
    """
    total_returns = terminal_prices / current_price - 1.0
    price_p10, price_p50, price_p90 = (
        float(value) for value in np.percentile(terminal_prices, PERCENTILES)
    )

    summary = ForecastSummary(
        model_used=model_used,
        model_version=MODEL_VERSION,
        horizon_days=settings.horizon_days,
        n_paths=settings.n_paths,
        seed=settings.seed,
        current_price=current_price,
        price_p10=price_p10,
        price_p50=price_p50,
        price_p90=price_p90,
        return_p10=price_p10 / current_price - 1.0,
        return_p50=price_p50 / current_price - 1.0,
        return_p90=price_p90 / current_price - 1.0,
        prob_positive=float(np.mean(total_returns > 0.0)),
        prob_loss_gt_5pct=float(np.mean(total_returns < -LOSS_THRESHOLD)),
        prob_low_vol=prob_low_vol,
        prob_high_vol=prob_high_vol,
    )

    log.info(
        "forecast model=%s n_paths=%d horizon=%d seed=%d return_p50=%.5f prob_positive=%.4f",
        summary.model_used,
        summary.n_paths,
        summary.horizon_days,
        summary.seed,
        summary.return_p50,
        summary.prob_positive,
    )
    return summary


def _validated_transition(P) -> np.ndarray:
    """A ``(k, k)`` left-stochastic matrix, or a loud failure at the point of use."""
    matrix = np.asarray(P, dtype=float)
    if matrix.shape != (K_REGIMES, K_REGIMES):
        raise ValueError(f"expected a ({K_REGIMES}, {K_REGIMES}) matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition matrix contains non-finite values")
    if not np.allclose(matrix.sum(axis=0), 1.0):
        raise ValueError(
            "transition matrix is not LEFT-stochastic — COLUMNS must sum to 1 "
            f"(architecture doc §5), got column sums {matrix.sum(axis=0).tolist()}"
        )
    return matrix


def _validated_scales(mus: np.ndarray, sigmas: np.ndarray) -> None:
    """Decimal-scale sanity. A sigma of 1.5 is a percent-scale parameter that skipped the rescale."""
    if not np.all(np.isfinite(mus)) or not np.all(np.isfinite(sigmas)):
        raise ValueError("mu/sigma must be finite before simulation")
    if np.any(sigmas <= 0.0):
        raise ValueError(f"sigmas must be positive, got {sigmas.tolist()}")
    if np.any(sigmas >= 1.0):
        raise ValueError(
            f"sigma {sigmas.tolist()} is at least 1.0 in DECIMAL scale, i.e. a 100% daily move — "
            "the percent-scale parameters were not divided by 100 (spec B-0)"
        )


def _validated_price(current_price: float) -> float:
    price = float(current_price)
    if not np.isfinite(price) or price <= 0.0:
        raise ValueError(f"current_price must be positive and finite, got {current_price}")
    return price
