"""Look-ahead / leakage tests (spec B-7). The tests that decide whether the backtest means
anything.

- ``test_tvtp_no_lookahead``: fit and forecast at origin T. Then corrupt every
  ``news_sentiment_3d`` value AFTER T with random values, refit and forecast again, and assert
  the ``ForecastSummary`` is bit-identical. If it changed, future news reached the fit — the
  ``shift(1)`` alignment in ``news_markov.py`` is wrong.
- Grep-level assertion: no reference to ``smoothed_marginal_probabilities`` anywhere under
  ``src/models/``. Smoothed probabilities use the full sample; reading them makes a backtest
  look excellent and mean nothing. A static check is used here because the numerical
  difference is easy to miss by eye.
- The same corrupt-the-future idea applied to prices: overwriting the closes and returns after T
  must not change the forecast at T on any of the three arms. Only the realized outcome may move.
- The training window itself, asserted against the clean rows through T computed by hand, on a
  frame whose post-T rows have been poisoned with an absurd value.

If the backtest ever comes back wildly good, these tests are the first thing to check.

WHY THE GREP CHECK IS A STRICT LITERAL SEARCH. The forbidden identifier appears nowhere under
``src/models/`` — not in code, and deliberately not in a docstring either, where those modules
spell the concept out in prose instead. That is what lets this test be one unambiguous string
search rather than a parser that has to decide which occurrences are "only a comment", which is
exactly the judgement a leak would hide behind.

WHAT THE B-3 SECTION PROVES, and why it is kept now that B-5 exists. The corrupt-the-future tests
show that future data does not REACH the fit; the B-3 tests show the lag is aligned CORRECTLY.
Those are different failures: an ``exog_tvtp`` built with ``shift(-1)`` and then sliced to T would
pass every corruption test in this file and still hand the model tomorrow's news. Row t must hold
YESTERDAY's news, and the row the lag consumes must leave ``endog`` and ``exog_tvtp`` the same
length (architecture doc §5).

WHY THE CORRUPTION TESTS DEMAND EXACT EQUALITY. ``ForecastSummary`` is compared with ``==``, not
with a tolerance, because a leak has no minimum size: any post-T value that reaches the fit changes
the fitted parameters, and any change in the parameters changes the simulated percentiles. A
tolerance would be a decision about how much leakage is acceptable, and the answer is none. This is
why every forecast gets its own seeded Generator — reproducibility is what makes exact equality a
legitimate assertion rather than a flaky one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import FitError, percent_returns
from src.models import backtest
from src.models.backtest import feature_dates, fit_arm, origin_window
from src.models.news_markov import LAG_DAYS, build_tvtp, fit_news_markov

MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"

#: The statsmodels attribute that leaks the future. Filtered probabilities use data through t;
#: smoothed probabilities use the whole sample, including everything after t.
FORBIDDEN_ATTRIBUTE = "smoothed_marginal_probabilities"

#: What the modeling layer may read instead (architecture doc §5).
REQUIRED_ATTRIBUTE = "filtered_marginal_probabilities"


def _model_sources() -> dict[str, str]:
    """Every modeling source, RECURSIVELY.

    rglob, not glob: a non-recursive scan silently exempts any future subpackage under
    ``src/models/``, and "the leak test passes because it never looked" is the worst way for this
    check to fail. Keys are paths relative to ``src/models`` so an offender in a subpackage is
    identifiable.
    """
    sources = {
        str(path.relative_to(MODELS_DIR).as_posix()): path.read_text(encoding="utf-8")
        for path in MODELS_DIR.rglob("*.py")
    }
    assert sources, f"no modeling sources found under {MODELS_DIR}"
    return sources


def test_no_smoothed_probabilities_under_src_models():
    """The grep test (spec B-7). One string, every file, no exceptions."""
    offenders = sorted(
        name for name, source in _model_sources().items() if FORBIDDEN_ATTRIBUTE in source
    )

    assert offenders == [], (
        f"{FORBIDDEN_ATTRIBUTE} appears in {offenders}. Smoothed probabilities incorporate "
        "observations after t, so any forecast built on them is a look-ahead leak "
        "(architecture doc §5)."
    )


def test_the_filtered_probabilities_are_actually_read():
    """Guard the guard: the grep test also passes if nothing reads any probabilities at all."""
    readers = sorted(
        name for name, source in _model_sources().items() if REQUIRED_ATTRIBUTE in source
    )

    assert "markov.py" in readers


# ------------------------------------------------------- B-3: the one-day TVTP lag


def test_build_tvtp_row_holds_yesterdays_news():
    """Row t of ``exog_tvtp`` carries the news from t-1, and the ones column is the intercept."""
    news = np.array([0.10, 0.20, 0.30, 0.40])

    exog = build_tvtp(news)

    assert exog.shape == (news.size - LAG_DAYS, 2)
    assert exog[:, 0] == pytest.approx(np.ones(3))
    assert exog[:, 1] == pytest.approx([0.10, 0.20, 0.30])


def test_build_tvtp_never_carries_same_day_news():
    """The leak, stated as an assertion: no row may hold the news of its own day.

    statsmodels builds the transition INTO t from row t. If row t held news[t], the model would
    know today's sentiment before deciding today's regime, and every backtest number after that
    is fiction. The series is strictly increasing so "yesterday" and "today" are never equal by
    coincidence.
    """
    news = np.arange(1.0, 11.0) / 10.0

    exog = build_tvtp(news)
    same_day = news[LAG_DAYS:]

    assert np.all(exog[:, 1] != same_day)
    assert exog[:, 1] == pytest.approx(news[:-LAG_DAYS])


def test_build_tvtp_matches_the_spec_expression():
    """Equivalence with the literal spec code: ``column_stack([ones, n.shift(1)])``, first row dropped."""
    series = pd.Series([0.0, -0.25, 0.5, 0.75, -1.0])

    lagged = series.shift(LAG_DAYS)
    expected = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])[LAG_DAYS:]

    assert build_tvtp(series) == pytest.approx(expected)


def test_fit_news_markov_drops_the_first_row_from_both(monkeypatch):
    """The joint drop (spec B-3). The C-e failure mode is a length mismatch, so lengths are asserted."""
    returns = np.arange(300.0) / 100.0
    news = np.linspace(-1.0, 1.0, 300)
    captured: dict = {}

    def recorder(endog, exog_tvtp=None, **kwargs):
        captured["endog"] = np.asarray(endog)
        captured["exog_tvtp"] = np.asarray(exog_tvtp)
        return "fit-result"

    monkeypatch.setattr("src.models.news_markov.fit_markov", recorder)

    assert fit_news_markov(returns, news) == "fit-result"
    assert captured["endog"] == pytest.approx(returns[LAG_DAYS:])
    assert captured["exog_tvtp"][:, 1] == pytest.approx(news[:-LAG_DAYS])
    assert captured["endog"].shape[0] == captured["exog_tvtp"].shape[0]


def test_fit_news_markov_rejects_misaligned_inputs():
    """Returns and news must already be row-aligned; guessing which end to trim is not an option."""
    with pytest.raises(FitError, match="same rows of daily_features"):
        fit_news_markov(np.zeros(300), np.zeros(299))


def test_build_tvtp_rejects_a_null_news_column():
    """A-4 writes 0 for a session with no news, never NULL, so a NaN here means the wrong column."""
    with pytest.raises(ValueError, match="non-finite"):
        build_tvtp(np.array([0.1, np.nan, 0.3]))


def test_fitted_model_c_carries_one_row_less_than_model_b(fitted_news_markov, fitted_markov):
    """The lag costs Model C its first observation — a property of the lag, not a different window."""
    assert fitted_news_markov.model.tvtp is True
    assert fitted_markov.model.tvtp is False
    assert fitted_news_markov.nobs == fitted_markov.nobs - LAG_DAYS


def test_model_c_is_not_charged_twice_for_the_row_its_lag_consumes():
    """A window of exactly ``min_obs`` rows must be fittable by Model C (spec B-0 vs B-3).

    Checking ``min_obs`` after the trim instead would refuse every minimum-length window and record
    a Model C failure at every short origin — a fallback rate that measures an off-by-one rather
    than the model.
    """
    with pytest.raises(FitError, match="refusing to fit on 59 observations"):
        fit_news_markov(np.zeros(59), np.zeros(59), min_obs=60)


# ------------------------------------------------------- B-5: corrupting the future


#: The origin every corruption test uses. Deep enough inside the frame that ~40 sessions sit after
#: it: corrupting five rows would be a weak test of a leak.
ORIGIN_INDEX = 100

#: Post-T rows are overwritten with this in the poisoned-frame test. Absurd on purpose — if one row
#: of it reached the fit, the fitted sigma could not be mistaken for a plausible one.
POISON = 1.0e6


def corrupted_after(frame, index: int, columns: list[str], values) -> "pd.DataFrame":
    """A COPY of ``frame`` whose ``columns`` are overwritten strictly after row ``index``."""
    poisoned = frame.copy()
    poisoned.loc[poisoned.index[index + 1 :], columns] = values
    return poisoned


def test_tvtp_no_lookahead(backtest_frame, backtest_cfg):
    """Corrupt every ``news_sentiment_3d`` after T; the forecast at T must not move (spec B-7).

    This is the test that decides whether Model C's backtest means anything. News is Model C's
    transition input, so if the ``shift(1)`` alignment were wrong — or if the window were sliced
    with ``<`` on the wrong side — post-T sentiment would enter ``exog_tvtp`` and the summary would
    change. Bit-identical, no tolerance.
    """
    dates = feature_dates(backtest_frame)
    origin = dates[ORIGIN_INDEX]
    future = slice(ORIGIN_INDEX + 1, None)

    rng = np.random.default_rng(20260810)
    poisoned = corrupted_after(
        backtest_frame,
        ORIGIN_INDEX,
        ["news_sentiment_3d"],
        rng.uniform(-1.0, 1.0, size=len(backtest_frame) - ORIGIN_INDEX - 1),
    )
    # Guard the guard: a corruption that changed nothing would make this test pass for free.
    assert not np.array_equal(
        poisoned["news_sentiment_3d"].to_numpy()[future],
        backtest_frame["news_sentiment_3d"].to_numpy()[future],
    )

    baseline = fit_arm(
        "news_markov",
        origin_window(backtest_frame, origin, dates=dates, horizon_days=5),
        backtest_cfg,
        min_obs=60,
    )
    refitted = fit_arm(
        "news_markov",
        origin_window(poisoned, origin, dates=dates, horizon_days=5),
        backtest_cfg,
        min_obs=60,
    )

    assert baseline.model_used == "news_markov"  # the TVTP path really was exercised
    assert refitted.summary == baseline.summary
    assert refitted.converged == baseline.converged


def test_corrupting_prices_after_t_does_not_move_the_forecast(backtest_frame, backtest_cfg):
    """The price variant of the same idea, on all three arms.

    Both ``close`` and ``log_return`` are corrupted, because in production one implies the other:
    ``log_return`` at T+1 is computed from the close at T+1. Only the realized outcome may depend on
    those rows, never the forecast.
    """
    dates = feature_dates(backtest_frame)
    origin = dates[ORIGIN_INDEX]
    poisoned = corrupted_after(backtest_frame, ORIGIN_INDEX, ["close", "log_return"], POISON)

    clean = origin_window(backtest_frame, origin, dates=dates, horizon_days=5)
    dirty = origin_window(poisoned, origin, dates=dates, horizon_days=5)

    for arm in ("news_markov", "markov", "gbm"):
        baseline = fit_arm(arm, clean, backtest_cfg, min_obs=60)
        refitted = fit_arm(arm, dirty, backtest_cfg, min_obs=60)
        assert refitted.summary == baseline.summary, arm

    # The outcome is the one thing that SHOULD have changed: it is measured after T by definition.
    assert dirty.realized_return != clean.realized_return


def test_the_training_window_never_contains_a_row_after_t(
    monkeypatch, backtest_frame, backtest_cfg
):
    """Asserted on the array the fit function received, against the rows through T computed by hand.

    Equality with the whole clean prefix is the assertion, not "no poison present": a window that
    stopped a row EARLY would also contain no poison, and that is a different bug with the same
    symptom in a weaker test.
    """
    dates = feature_dates(backtest_frame)
    origin = dates[ORIGIN_INDEX]
    poisoned = corrupted_after(
        backtest_frame, ORIGIN_INDEX, ["close", "log_return", "news_sentiment_3d"], POISON
    )

    received: dict = {}
    real_markov = backtest.fit_markov

    def spy(returns_pct, **kwargs):
        received["endog"] = np.array(returns_pct, copy=True)
        return real_markov(returns_pct, **kwargs)

    monkeypatch.setattr(backtest, "fit_markov", spy)

    window = origin_window(poisoned, origin, dates=dates, horizon_days=5)
    fit_arm("markov", window, backtest_cfg, min_obs=60)

    expected = percent_returns(backtest_frame["log_return"].to_numpy()[: ORIGIN_INDEX + 1])
    assert received["endog"] == pytest.approx(expected)
    assert received["endog"].size == ORIGIN_INDEX
    # The exog input is trimmed from the same rows, so it must be poison-free too.
    assert window.news == pytest.approx(
        backtest_frame["news_sentiment_3d"].to_numpy()[1 : ORIGIN_INDEX + 1]
    )
