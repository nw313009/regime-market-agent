"""Model C — News-Markov with time-varying transition probabilities (spec B-3).

Model C is Model B plus news sentiment as an exogenous driver of the transition
probabilities. It answers one question, without presuming the answer is yes:

    Does recent financial news improve five-trading-day probabilistic forecasting beyond
    price-driven regime switching alone?

No LLM-generated feature ever enters this path.

exog_tvtp construction — THE alignment rule::

    n = df["news_sentiment_3d"].shift(1)          # lag one trading day
    exog_tvtp = np.column_stack([np.ones(len(n)), n])
    # drop the first row jointly with endog so the lengths match

statsmodels uses ``exog_tvtp`` row t to build the transition INTO t, i.e. the transition
from state t-1 to state t. Shifting by one day is what makes "news known at t affects the
t -> t+1 transition" actually true, instead of letting the model peek at same-day news.

The column of ones is mandatory: it is the intercept of the transition model.

A "lengths differ" error on ``exog_tvtp`` means the ``shift(1)`` row was not dropped
jointly with ``endog``.

The no-lookahead test in ``tests/test_no_lookahead.py`` is what proves the alignment.

WHY THE DROP IS DONE HERE AND NOT BY THE CALLER. ``shift(1)`` leaves a NaN in the first row,
which has no predecessor to borrow news from. Dropping it from the exog alone shortens the exog
and statsmodels raises; dropping it from neither leaves a NaN in the fit. :func:`fit_news_markov`
therefore trims both in one place, and the two arrays it was handed must be aligned row-for-row
before it is called — same ticker, same trade dates, same warm-up already dropped from both.
That precondition is checked, not assumed.

PARITY, HONESTLY STATED. The B-5 parity rule fits all three models on the identical training
window. Model C's ``endog`` is nonetheless one observation shorter, because the lag consumes the
first row: there is no news from the day before the window started. That is a property of the lag,
not a different window, and it is the reason Model C is not simply "Model B with extra columns".

CONSEQUENTLY, ``min_obs`` IS CHECKED BEFORE THE LAG. The B-0 refusal is about the training WINDOW
("refuse to fit with fewer than ``min_train_days`` observations"), and the lag is an internal
detail of this model. Checking it after the trim would refuse every window of exactly
``min_train_days`` rows and record a Model C failure at every minimum-length backtest origin — a
fallback rate that measures an off-by-one rather than the model.
"""

from __future__ import annotations

import logging

import numpy as np

from src.models import MIN_TRAIN_DAYS, FitError, validated_returns
from src.models.markov import SEARCH_REPS, START_SEARCH_SEED, fit_markov

__all__ = ["LAG_DAYS", "MODEL_NAME", "NEWS_COLUMN", "build_tvtp", "fit_news_markov"]

log = logging.getLogger(__name__)

#: Recorded as ``model_used`` when the ladder lands on this rung.
MODEL_NAME = "news_markov"

#: The ``silver.daily_features`` column that drives the transitions (spec A-4).
NEWS_COLUMN = "news_sentiment_3d"

#: One trading day. Not configurable: this is the alignment rule (architecture doc §5), and a
#: "tunable" lag is a lookahead bug with a config key in front of it.
LAG_DAYS = 1


def build_tvtp(news_series) -> np.ndarray:
    """``[ones, news.shift(1)]`` with the first row dropped — the transition input for Model C.

    Returns an ``(n - 1, 2)`` array whose row ``t`` holds ``[1.0, news[t - 1]]``, so it lines up
    with ``endog[1:]``. Because statsmodels builds the transition INTO t from row t, that makes
    the transition into day t depend on news that was already public on day t-1.

    ``news_sentiment_3d`` is never NULL by construction (A-4 treats a missing lag as 0), so a
    non-finite value here means the column was not the one A-4 wrote and it raises rather than
    being imputed.
    """
    news = np.asarray(news_series, dtype=float)
    if news.ndim != 1:
        raise ValueError(f"{NEWS_COLUMN} must be 1-D, got shape {news.shape}")
    if news.size <= LAG_DAYS:
        raise ValueError(
            f"{NEWS_COLUMN} has {news.size} rows; at least {LAG_DAYS + 1} are needed to form a "
            "lagged transition input"
        )
    if not np.all(np.isfinite(news)):
        raise ValueError(
            f"{NEWS_COLUMN} contains non-finite values; A-4 writes 0 for a session with no news, "
            "never NULL"
        )

    lagged = news[:-LAG_DAYS]
    return np.column_stack([np.ones(lagged.size), lagged])


def fit_news_markov(
    returns_pct,
    news_series,
    *,
    min_obs: int = MIN_TRAIN_DAYS,
    search_reps: int = SEARCH_REPS,
    search_seed: int = START_SEARCH_SEED,
):
    """Fit Model C: ``fit_markov`` with the lagged news transition input.

    ``returns_pct`` and ``news_series`` must be the same length and already aligned row-for-row
    (same ticker, same trade dates, warm-up dropped from both). The first row of each is consumed
    by the lag.

    Delegates the whole fit — the specification, ``search_reps``, the non-finite guard and both
    degeneracy checks — to :func:`~src.models.markov.fit_markov`, so Models B and C cannot drift
    apart in anything except the transition input. That is the only difference between them, and
    it is the difference the evaluation is measuring.
    """
    returns = validated_returns(returns_pct, min_obs=min_obs, model=MODEL_NAME)
    news = np.asarray(news_series, dtype=float)

    if returns.shape != news.shape:
        raise FitError(
            f"{MODEL_NAME}: returns_pct has {returns.shape[0]} rows and {NEWS_COLUMN} has "
            f"{news.shape[0]} — they must be sliced from the same rows of daily_features before "
            "the lag is applied (spec B-3)"
        )

    exog_tvtp = build_tvtp(news)
    endog = returns[LAG_DAYS:]

    log.info(
        "%s fitting n=%d (%d after the %d-day lag)",
        MODEL_NAME,
        returns.size,
        endog.size,
        LAG_DAYS,
    )
    # The window was already checked against min_obs above; the delegated check must not charge
    # Model C twice for the row its own lag consumed.
    return fit_markov(
        endog,
        exog_tvtp=exog_tvtp,
        min_obs=max(min_obs - LAG_DAYS, 0),
        search_reps=search_reps,
        search_seed=search_seed,
    )
