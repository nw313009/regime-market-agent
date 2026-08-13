"""Walk-forward backtest (spec B-5). A core deliverable, not a stretch goal.

Not part of the daily job: this runs on demand as a separate job/notebook, and the app
reads the results out of ``gold.backtest_metrics``.

Origins: weekly, over the last ``cfg.backtest.n_weeks`` weeks, per ticker, each origin
requiring at least ``cfg.backtest.min_train_days`` training rows ending at T.

At each origin T:

1. Build features and ``exog_tvtp`` using data through T only.
2. Fit the ladder C -> B -> A, recording ``model_used`` and the failure reason.
3. Forecast T+5 through ``monte_carlo.run_forecast``, decaying news from ``N_T``.
4. Score against the realized 5-day return.

Parity rule: at each origin all three models fit on the identical training window.

No future price or news observation may enter fitting, regime probabilities, features,
transition variables, or initialization.

Metrics, pooled across tickers and always reported with n:

- Brier score on P(R5 > 0)
- MAE of the median return
- 80% prediction-interval coverage
- per-model fallback rate (how often C failed to converge)

Writes one row per ``(origin, ticker, model)`` to ``gold.backtest_metrics``, plus a pooled
summary table.

A backtest that looks wildly good is a leak: check the smoothed-probabilities grep test and
the TVTP lag test before believing it.

THREE ARMS, EACH WITH ITS OWN LADDER. "One row per (origin, ticker, model)" and "fit the ladder
C -> B -> A" are two different requirements and this module implements both. Each model is
evaluated as its own ARM, because the Model Evaluation page compares GBM against Markov against
News-Markov; and each arm descends the ladder from its own rung, because that is what production
does when a fit fails. So ``model`` is the arm that was asked for and ``model_used`` is the rung
that answered, and the per-model fallback rate is simply how often those two differ. Arm A cannot
fall back — it is the bottom — which is exactly why its fallback rate is 0 and not a missing value.

WHY EVERY FORECAST GETS A FRESH GENERATOR. ``rng_for(cfg)`` is called per (origin, arm), so all
three arms at one origin draw the SAME uniforms. That is common random numbers: the arms are
compared on identical noise, so a Brier difference between them is a difference in the models
rather than in the draws. It also makes any single origin reproducible in isolation, which is what
lets the no-lookahead test demand a bit-identical summary instead of a tolerance.

THE ONLY LINE THAT PREVENTS LEAKAGE is the slice in :func:`origin_window`: the training frame is
``frame.iloc[: index_of(T) + 1]``. Positional slicing on a frame validated as strictly increasing
in ``trade_date`` is used rather than a date comparison because it makes "no row after T" a
structural property of the code instead of a property of a predicate someone might loosen later.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from src.models import MIN_TRAIN_DAYS, FitError, percent_returns, warmup_offset
from src.models.gbm import MODEL_NAME as GBM
from src.models.gbm import fit_gbm
from src.models.markov import MODEL_NAME as MARKOV
from src.models.markov import fit_markov, sort_regimes
from src.models.monte_carlo import (
    ForecastSummary,
    forecast_config,
    rng_for,
    run_forecast,
    run_gbm_forecast,
)
from src.models.news_markov import MODEL_NAME as NEWS_MARKOV
from src.models.news_markov import fit_news_markov

__all__ = [
    "FEATURE_COLUMNS",
    "LADDERS",
    "MODEL_ARMS",
    "TASK_NAME",
    "ArmFit",
    "BacktestResult",
    "BacktestRow",
    "OriginWindow",
    "PooledRow",
    "fit_arm",
    "origin_window",
    "pooled_summary",
    "production_window",
    "run_backtest",
    "score_forecast",
    "weekly_origins",
]

log = logging.getLogger(__name__)

#: Ledger task name (``bronze.ingestion_runs.task``). On demand, not part of the daily workflow.
TASK_NAME = "run_backtest"

#: The columns this module reads out of ``silver.daily_features``.
FEATURE_COLUMNS = ("trade_date", "close", "log_return", "news_sentiment_3d")

#: The three evaluated arms, richest first, which is also ladder order.
MODEL_ARMS = (NEWS_MARKOV, MARKOV, GBM)

#: What each arm descends through on a ``FitError`` (architecture doc §5).
LADDERS: Mapping[str, tuple[str, ...]] = {
    NEWS_MARKOV: (NEWS_MARKOV, MARKOV, GBM),
    MARKOV: (MARKOV, GBM),
    GBM: (GBM,),
}


@dataclass(frozen=True, eq=False)
class OriginWindow:
    """Everything known at origin T, and nothing that was not.

    ``eq=False`` because two fields are arrays. Built once per origin and shared by all three
    arms — that sharing IS the parity rule, expressed as a single object rather than as three
    calls that must be kept in step.
    """

    origin: date
    #: Percent log returns of the training window, warm-up dropped (spec B-0).
    returns_pct: np.ndarray
    #: ``news_sentiment_3d`` over the SAME rows, before Model C applies its one-day lag.
    news: np.ndarray
    #: Close at T. The forecast starts here.
    current_price: float
    #: ``N_T``, the news signal at T, which the simulation decays over the horizon.
    current_news: float
    #: The realized horizon return, used for scoring only — never for fitting.
    realized_return: float

    @property
    def n_train(self) -> int:
        return int(self.returns_pct.size)


@dataclass(frozen=True, eq=False)
class ArmFit:
    """One arm's forecast at one origin, plus how far down the ladder it had to go."""

    model: str
    model_used: str
    summary: ForecastSummary
    #: Optimizer convergence, or ``None`` for GBM, which has no optimizer to converge.
    converged: bool | None
    #: Recorded reasons for every rung that failed above ``model_used``, or ``None``.
    failure_reason: str | None
    #: The rung's ``SortedParams``, or ``None`` when GBM answered — it has no regimes. Carried
    #: because the daily ``fit_models`` task (C-6) writes ``gold.regime_states`` from the SAME fit
    #: that produced the forecast. Refitting to recover the parameters would be a second MLE with
    #: its own optimizer path, and the two tables would then describe two different models.
    sorted_params: Any | None = None

    @property
    def fell_back(self) -> bool:
        return self.model_used != self.model


@dataclass(frozen=True)
class BacktestRow:
    """One row of ``gold.backtest_metrics``: one (origin, ticker, model) evaluation.

    ``mae`` holds a single ABSOLUTE ERROR at this grain; the pooled summary averages them into a
    mean absolute error. The column keeps the name spec B-6 gives it.

    ``realized_return``, ``return_p50`` and ``prob_positive`` are stored beyond B-6's column list
    on purpose: a Brier score with no record of the probability and the outcome it came from
    cannot be audited or recomputed, and this table is the evidence behind a published verdict.
    """

    origin_date: date
    ticker: str
    model: str
    brier: float
    mae: float
    covered_80: bool
    model_used: str
    converged: bool | None
    failure_reason: str | None
    realized_return: float
    return_p50: float
    prob_positive: float

    def as_row(self) -> dict:
        return {
            "origin_date": self.origin_date,
            "ticker": self.ticker,
            "model": self.model,
            "brier": float(self.brier),
            "mae": float(self.mae),
            "covered_80": bool(self.covered_80),
            "model_used": self.model_used,
            "converged": self.converged,
            "failure_reason": self.failure_reason,
            "realized_return": float(self.realized_return),
            "return_p50": float(self.return_p50),
            "prob_positive": float(self.prob_positive),
        }


@dataclass(frozen=True)
class PooledRow:
    """One row of the pooled summary: a model's scores across every origin and ticker.

    ``n`` is a first-class output, not a footnote. Twenty-six weekly origins on five tickers is
    130 forecasts per model, which is a small sample for a Brier difference — the Model Evaluation
    page has to show n so a reader can judge the comparison, and "no meaningful improvement
    detected at this sample size" is a legitimate verdict (spec A2).
    """

    model: str
    n: int
    n_tickers: int
    brier: float
    mae: float
    coverage_80: float
    fallback_rate: float

    def as_row(self, computed_at: datetime) -> dict:
        return {
            "model": self.model,
            "n": int(self.n),
            "n_tickers": int(self.n_tickers),
            "brier": float(self.brier),
            "mae": float(self.mae),
            "coverage_80": float(self.coverage_80),
            "fallback_rate": float(self.fallback_rate),
            "computed_at": computed_at,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Per-origin rows, the pooled summary, and what was skipped.

    Skips are carried rather than logged and forgotten: a silently shorter ``n`` is the difference
    between "Model C is no better" and "Model C was never evaluated here".
    """

    rows: tuple[BacktestRow, ...]
    pooled: tuple[PooledRow, ...]
    skipped: tuple[dict, ...]


# ------------------------------------------------------------------ origins and windows


def feature_dates(frame: Any) -> list[date]:
    """The frame's ``trade_date`` column as plain dates, validated strictly increasing.

    Positional slicing is what keeps the future out of a training window, and positional slicing
    is only safe on a sorted frame. The caller crossing the Spark boundary orders by ``trade_date``
    (spec C-b); this checks that it actually did.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"daily_features frame is missing {missing}")

    dates = [_as_date(value) for value in frame["trade_date"].tolist()]
    if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
        raise ValueError(
            "daily_features frame must be sorted by trade_date, strictly increasing — the "
            "backtest slices training windows positionally"
        )
    return dates


def weekly_origins(
    frame: Any,
    *,
    dates: Sequence[date] | None = None,
    n_weeks: int,
    min_train_days: int,
    horizon_days: int,
) -> list[date]:
    """The last ``n_weeks`` weekly origins that can be both fitted and scored (spec B-5).

    An origin is eligible when it has at least ``min_train_days`` USABLE training returns at or
    before it — warm-up rows carry a NULL ``log_return`` and are not training data — and when the
    frame extends ``horizon_days`` sessions beyond it, since an origin whose outcome has not
    happened yet cannot be scored and must not be counted in ``n``.

    "Weekly" is the LAST eligible session of each ISO week, not every fifth row: a holiday week
    has four sessions, and stepping by row count would drift the origins away from week ends and
    make two tickers with different histories land on different days.
    """
    if n_weeks <= 0:
        raise ValueError(f"n_weeks must be positive, got {n_weeks}")

    trade_dates = list(dates) if dates is not None else feature_dates(frame)
    usable = np.cumsum(~np.isnan(np.asarray(frame["log_return"].tolist(), dtype=float)))
    last_index = len(trade_dates) - 1

    by_week: dict[tuple[int, int], int] = {}
    for index, trade_date in enumerate(trade_dates):
        if usable[index] < min_train_days or index + horizon_days > last_index:
            continue
        iso = trade_date.isocalendar()
        by_week[(iso[0], iso[1])] = index

    return [trade_dates[index] for index in sorted(by_week.values())[-n_weeks:]]


def origin_window(
    frame: Any,
    origin: date,
    *,
    dates: Sequence[date] | None = None,
    horizon_days: int,
) -> OriginWindow:
    """Build the training window and the realized outcome for one origin.

    The training slice ends AT T inclusive and the realized return is read ``horizon_days``
    sessions later, from the full frame. Those are the only two places this function touches the
    frame, and they are deliberately the only two: everything an arm sees comes from the returned
    object, so no arm can reach past T even by accident.
    """
    trade_dates = list(dates) if dates is not None else feature_dates(frame)
    try:
        index = trade_dates.index(origin)
    except ValueError as exc:
        raise ValueError(f"origin {origin} is not a session in this frame") from exc
    if index + horizon_days > len(trade_dates) - 1:
        raise ValueError(
            f"origin {origin} has fewer than {horizon_days} sessions after it, so its outcome "
            "cannot be scored"
        )

    return _window_at(frame, index, trade_dates, outcome_index=index + horizon_days)


def production_window(frame: Any, *, dates: Sequence[date] | None = None) -> OriginWindow:
    """The window at the LAST session in the frame: the daily ``fit_models`` fit (spec A1 step 5).

    The same construction as :func:`origin_window` with one difference — there is no outcome, so
    ``realized_return`` is NaN and the window must never be scored. That is the whole distinction
    between the backtest and production: an origin is a day whose future has already happened, and
    today is not.

    It lives here, next to the window the backtest uses and the ladder that consumes it, because a
    second implementation of "the training window" in the Spark layer is exactly how a leak or an
    off-by-one row gets in on the production side without a single backtest test noticing.
    """
    trade_dates = list(dates) if dates is not None else feature_dates(frame)
    if not trade_dates:
        raise ValueError("cannot build a window from an empty daily_features frame")
    return _window_at(frame, len(trade_dates) - 1, trade_dates, outcome_index=None)


def _window_at(
    frame: Any,
    index: int,
    trade_dates: Sequence[date],
    *,
    outcome_index: int | None,
) -> OriginWindow:
    """Slice the training window ending at ``index`` and read the outcome, if there is one."""
    window = frame.iloc[: index + 1]
    log_return = np.asarray(window["log_return"].tolist(), dtype=float)
    news_column = np.asarray(window["news_sentiment_3d"].tolist(), dtype=float)
    closes = np.asarray(frame["close"].tolist(), dtype=float)

    returns_pct = percent_returns(log_return)
    # The SAME trim, from the same helper, so news cannot slide against returns by a row.
    news = news_column[warmup_offset(log_return) :]

    origin = trade_dates[index]
    current_price = float(closes[index])
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"close at origin {origin} is not a usable price: {current_price}")

    realized = (
        float("nan")
        if outcome_index is None
        else float(closes[outcome_index] / current_price - 1.0)
    )

    return OriginWindow(
        origin=origin,
        returns_pct=returns_pct,
        news=news,
        current_price=current_price,
        current_news=float(news_column[-1]),
        realized_return=realized,
    )


# ------------------------------------------------------------------ the ladder


def fit_arm(
    arm: str,
    window: OriginWindow,
    cfg: Mapping,
    *,
    min_obs: int = MIN_TRAIN_DAYS,
) -> ArmFit | None:
    """Fit and forecast one arm, descending its ladder on ``FitError``.

    Returns ``None`` when every rung failed, which is a reportable outcome rather than an
    exception: one unfittable window must not abort a 26-week backtest, and the skip is recorded
    so ``n`` stays honest.
    """
    if arm not in LADDERS:
        raise ValueError(f"unknown model arm {arm!r}, expected one of {sorted(LADDERS)}")

    failures: list[str] = []
    for rung in LADDERS[arm]:
        try:
            summary, converged, sorted_params = _forecast_rung(rung, window, cfg, min_obs=min_obs)
        except FitError as exc:
            failures.append(f"{rung}: {exc}")
            log.info("origin=%s arm=%s rung=%s failed: %s", window.origin, arm, rung, exc)
            continue
        return ArmFit(
            model=arm,
            model_used=rung,
            summary=summary,
            converged=converged,
            failure_reason="; ".join(failures) or None,
            sorted_params=sorted_params,
        )

    log.warning("origin=%s arm=%s produced no forecast: %s", window.origin, arm, "; ".join(failures))
    return None


def _forecast_rung(
    rung: str,
    window: OriginWindow,
    cfg: Mapping,
    *,
    min_obs: int,
) -> tuple[ForecastSummary, bool | None, Any | None]:
    """Fit one rung on the shared window and forecast the horizon from it.

    Returns the forecast, the optimizer's convergence flag, and the sorted regime parameters —
    the last of which is ``None`` for GBM and is what ``fit_models`` turns into a
    ``gold.regime_states`` row.
    """
    rng = rng_for(cfg)  # fresh Generator per forecast run: common random numbers across arms

    if rung == GBM:
        params = fit_gbm(window.returns_pct, min_obs=min_obs)
        return run_gbm_forecast(params, window.current_price, cfg, rng), None, None

    if rung == MARKOV:
        res = fit_markov(window.returns_pct, min_obs=min_obs)
        sorted_params = sort_regimes(res)
        return (
            run_forecast(
                sorted_params, None, window.current_price, window.current_news, cfg, rng
            ),
            sorted_params.converged,
            sorted_params,
        )

    res = fit_news_markov(window.returns_pct, window.news, min_obs=min_obs)
    sorted_params = sort_regimes(res)
    return (
        run_forecast(sorted_params, res, window.current_price, window.current_news, cfg, rng),
        sorted_params.converged,
        sorted_params,
    )


# ------------------------------------------------------------------ scoring


def score_forecast(summary: ForecastSummary, realized_return: float) -> dict:
    """Score one forecast against one realized return (spec B-5).

    - ``brier``: squared error of P(R5 > 0) against the outcome, so 0 is perfect and 0.25 is the
      score of an honest "no idea". A probability forecast has to be scored as a probability;
      accuracy on a directional call would reward overconfidence.
    - ``mae``: absolute error of the median return. At this grain it is one error; pooled, it is
      the mean absolute error.
    - ``covered_80``: whether the outcome fell inside [P10, P90]. Inclusive at both ends, and a
      boolean rather than a distance, because coverage is the property being measured — an 80%
      interval should contain the outcome about 80% of the time, and both under- and
      over-coverage are failures.
    """
    realized = float(realized_return)
    if not np.isfinite(realized):
        raise ValueError(f"realized_return must be finite, got {realized_return}")

    outcome = 1.0 if realized > 0.0 else 0.0
    return {
        "brier": float((summary.prob_positive - outcome) ** 2),
        "mae": float(abs(summary.return_p50 - realized)),
        "covered_80": bool(summary.return_p10 <= realized <= summary.return_p90),
    }


def pooled_summary(rows: Sequence[BacktestRow]) -> tuple[PooledRow, ...]:
    """Pool the per-origin rows per model, across tickers, always with n (spec B-5).

    Arms with no rows are omitted rather than reported as zeros: an arm that was never scored has
    no Brier score, and printing 0.0 for it would read as a perfect one.
    """
    pooled: list[PooledRow] = []
    for model in MODEL_ARMS:
        scored = [row for row in rows if row.model == model]
        if not scored:
            continue
        pooled.append(
            PooledRow(
                model=model,
                n=len(scored),
                n_tickers=len({row.ticker for row in scored}),
                brier=float(np.mean([row.brier for row in scored])),
                mae=float(np.mean([row.mae for row in scored])),
                coverage_80=float(np.mean([row.covered_80 for row in scored])),
                fallback_rate=float(
                    np.mean([row.model_used != row.model for row in scored])
                ),
            )
        )
    return tuple(pooled)


# ------------------------------------------------------------------ the walk-forward run


def backtest_ticker(
    ticker: str,
    frame: Any,
    cfg: Mapping,
) -> tuple[list[BacktestRow], list[dict]]:
    """Walk one ticker's origins, fitting all three arms on each shared window."""
    settings = forecast_config(cfg)
    backtest_cfg = cfg["backtest"]
    min_train_days = int(backtest_cfg["min_train_days"])

    dates = feature_dates(frame)
    origins = weekly_origins(
        frame,
        dates=dates,
        n_weeks=int(backtest_cfg["n_weeks"]),
        min_train_days=min_train_days,
        horizon_days=settings.horizon_days,
    )
    if not origins:
        log.warning(
            "ticker=%s has no eligible origins: %d sessions, %d required plus a %d-session outcome",
            ticker,
            len(dates),
            min_train_days,
            settings.horizon_days,
        )

    rows: list[BacktestRow] = []
    skipped: list[dict] = []

    for origin in origins:
        window = origin_window(frame, origin, dates=dates, horizon_days=settings.horizon_days)
        for arm in MODEL_ARMS:
            fit = fit_arm(arm, window, cfg, min_obs=min_train_days)
            if fit is None:
                skipped.append({"ticker": ticker, "origin_date": origin, "model": arm})
                continue
            rows.append(
                BacktestRow(
                    origin_date=origin,
                    ticker=ticker,
                    model=fit.model,
                    model_used=fit.model_used,
                    converged=fit.converged,
                    failure_reason=fit.failure_reason,
                    realized_return=window.realized_return,
                    return_p50=fit.summary.return_p50,
                    prob_positive=fit.summary.prob_positive,
                    **score_forecast(fit.summary, window.realized_return),
                )
            )

    log.info(
        "backtest ticker=%s origins=%d rows=%d skipped=%d",
        ticker,
        len(origins),
        len(rows),
        len(skipped),
    )
    return rows, skipped


def run_backtest(frames: Mapping[str, Any], cfg: Mapping) -> BacktestResult:
    """Walk every ticker and pool the results (spec B-5).

    ``frames`` maps ticker to its ``silver.daily_features`` rows as a pandas frame, ordered by
    ``trade_date`` — one ``.toPandas()`` per ticker at the Spark boundary (spec C-b). Spark stays
    on the caller's side of that boundary; nothing in this module knows it exists.
    """
    rows: list[BacktestRow] = []
    skipped: list[dict] = []

    for ticker in sorted(frames):
        ticker_rows, ticker_skipped = backtest_ticker(ticker, frames[ticker], cfg)
        rows.extend(ticker_rows)
        skipped.extend(ticker_skipped)

    pooled = pooled_summary(rows)
    for row in pooled:
        log.info(
            "pooled model=%s n=%d tickers=%d brier=%.4f mae=%.4f coverage=%.3f fallback=%.3f",
            row.model,
            row.n,
            row.n_tickers,
            row.brier,
            row.mae,
            row.coverage_80,
            row.fallback_rate,
        )
    return BacktestResult(rows=tuple(rows), pooled=pooled, skipped=tuple(skipped))


def _as_date(value: Any) -> date:
    """Normalize a pandas/numpy/py date-like to ``datetime.date``.

    ``.toPandas()`` renders a Delta DATE column as ``datetime64``, whose elements arrive as
    ``Timestamp`` (a ``datetime`` subclass), while a hand-built frame may hold plain ``date``
    objects. Both are normalized here so the rest of the module compares like with like.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[s]").astype(datetime).date()
    converted = getattr(value, "to_pydatetime", None)
    if converted is not None:
        return converted().date()
    raise TypeError(f"trade_date must be a date, got {type(value).__name__}: {value!r}")
