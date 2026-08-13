"""Daily ``fit_models`` task (spec A1 step 5, wired at C-6).

Per ticker: read ``silver.daily_features`` through one ``.toPandas()``, fit the C -> B -> A ladder
on the FULL available history ending at the last session, and write both gold tables the fit
produces — ``gold.regime_states`` from the sorted regime parameters and ``gold.forecast_runs`` from
the Monte Carlo run.

ONE TASK, TWO TABLES, ONE FIT. Spec A1 lists ``fit_models`` and ``run_forecasts`` as separate
steps; this module does both, because the forecast is simulated FROM the fitted parameters. Split
across two workflow tasks they would have to either refit (two MLE runs whose optimizer paths can
land in different local optima, so the published regime state would describe a different model
than the published forecast) or serialize a fitted statsmodels result between tasks. Neither is
worth a task boundary. ``bronze.ingestion_runs`` therefore carries one ``fit_models`` row per run
covering both writes, which is what :func:`write_gold` is built for.

WHAT THIS MODULE DOES NOT DO: no statistics. Every number comes from ``src/models/`` — the window
from :func:`production_window`, the ladder from :func:`fit_arm`, the row shapes from
``src/pipelines``. Spark appears here and only here (hard rule 3), on either side of the
``.toPandas()`` boundary.

A GBM DAY WRITES NO REGIME ROW. When both Markov rungs fail and the ladder lands on A, there are
no regimes to publish, and ``gold.regime_states`` gets nothing for that ticker rather than a row of
zeros. The forecast is still written, with ``model_used = 'gbm'`` and NULL regime probabilities, so
the app can say what it actually has. Yesterday's regime row stays in place, distinguishable by its
``as_of_date``.

A ticker whose every rung failed is reported in the summary and skipped, never written as a
partial row: one unfittable ticker must not take the other four down with it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.models.backtest import FEATURE_COLUMNS, MODEL_ARMS, fit_arm, production_window
from src.pipelines import (
    GOLD_FORECAST_RUNS,
    GOLD_REGIME_STATES,
    forecast_run_row,
    qualified,
    quote_identifier,
    regime_state_row,
    require_table,
    utc_now,
    write_gold,
)

__all__ = [
    "FEATURES_TABLE",
    "PRODUCTION_ARM",
    "TASK_NAME",
    "TickerFit",
    "fit_ticker",
    "main",
    "read_features",
    "tickers_with_features",
]

log = logging.getLogger(__name__)

#: Ledger task name (``bronze.ingestion_runs.task``).
TASK_NAME = "fit_models"

#: The source of every number this task fits on.
FEATURES_TABLE = "silver.daily_features"

#: Production always asks for the richest arm and lets the ladder answer. ``model_used`` records
#: which rung did, so a fallback is visible in gold rather than inferred from a log.
PRODUCTION_ARM = MODEL_ARMS[0]


@dataclass
class TickerFit:
    """What one ticker's fit produced: up to one row for each gold table, or a reason it did not."""

    ticker: str
    as_of_date: date | None = None
    model_used: str | None = None
    regime_row: dict | None = None
    forecast_row: dict | None = None
    skipped_reason: str | None = None
    sessions: int = 0

    @property
    def ok(self) -> bool:
        return self.forecast_row is not None


@dataclass
class FitSummary:
    """The task's own report, for notebook display and for the workflow log."""

    task: str = TASK_NAME
    run_id: str | None = None
    fits: list[TickerFit] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    rows_written: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "run_id": self.run_id,
            "tickers": [fit.ticker for fit in self.fits if fit.ok],
            "model_used": {fit.ticker: fit.model_used for fit in self.fits if fit.ok},
            "skipped": list(self.skipped),
            "rows_written": dict(self.rows_written),
        }


def tickers_with_features(spark: Any, catalog: str) -> list[str]:
    """Every ticker ``silver.daily_features`` holds.

    The universe is taken from the feature table rather than from ``config.tickers.seed`` plus the
    watchlist, for one reason: this task cannot read the watchlist. The watchlist lives in Lakebase
    and ``src/database/lakebase.py`` imports psycopg, which SIGABRTs a serverless kernel at import
    (see ``requirements-databricks.txt``). Reading the feature table gets the same answer one step
    later — ingestion already resolved seed-plus-watchlist, so a ticker added through the agent
    appears here as soon as it has been ingested and built — and it never fits a ticker that has no
    data, which is what asking the watchlist directly would do on the day a ticker is added.
    """
    rows = spark.sql(
        f"SELECT DISTINCT ticker FROM {qualified(catalog, FEATURES_TABLE)} ORDER BY ticker"
    ).collect()
    return [str(row["ticker"]) for row in rows]


def read_features(spark: Any, catalog: str, ticker: str) -> Any:
    """One ticker of ``silver.daily_features`` as a pandas frame (spec C-b).

    THE SPARK BOUNDARY. Ordered by ``trade_date`` because the modeling layer slices positionally,
    projected to the columns the models read, and the ticker travels as a parameter marker rather
    than inside the SQL text.
    """
    columns = ", ".join(quote_identifier(column) for column in FEATURE_COLUMNS)
    return spark.sql(
        f"SELECT {columns} FROM {qualified(catalog, FEATURES_TABLE)} "
        "WHERE ticker = :ticker ORDER BY trade_date",
        args={"ticker": ticker},
    ).toPandas()


def fit_ticker(ticker: str, frame: Any, config: Mapping) -> TickerFit:
    """Fit the ladder at the last session and build the gold rows. PURE — no Spark, no I/O.

    Returns a :class:`TickerFit` in every case, including failure, so the caller can report what
    happened per ticker instead of catching exceptions to find out.
    """
    fit = TickerFit(ticker=ticker, sessions=0 if frame is None else len(frame))

    if fit.sessions == 0:
        fit.skipped_reason = "no rows in silver.daily_features"
        return fit

    min_obs = int(config["backtest"]["min_train_days"])
    try:
        window = production_window(frame)
    except ValueError as exc:
        fit.skipped_reason = f"unusable window: {exc}"
        return fit

    arm = fit_arm(PRODUCTION_ARM, window, config, min_obs=min_obs)
    if arm is None:
        fit.skipped_reason = "every rung of the ladder failed"
        return fit

    fit.as_of_date = window.origin
    fit.model_used = arm.model_used
    fit.forecast_row = forecast_run_row(arm.summary, ticker=ticker, as_of_date=window.origin)

    if arm.sorted_params is not None:
        fit.regime_row = regime_state_row(
            arm.sorted_params,
            ticker=ticker,
            as_of_date=window.origin,
            current_news_signal=window.current_news,
            model_used=arm.model_used,
            model_version=arm.summary.model_version,
        )
    else:
        log.info(
            "%s %s: ladder landed on %s, which has no regimes — no regime_states row",
            TASK_NAME,
            ticker,
            arm.model_used,
        )

    if arm.failure_reason:
        log.warning("%s %s: fell back to %s (%s)", TASK_NAME, ticker, arm.model_used, arm.failure_reason)
    return fit


def main(
    spark: Any,
    config: Mapping,
    *,
    tickers: Sequence[str] | None = None,
    catalog: str | None = None,
) -> dict:
    """Fit every ticker and MERGE the two gold tables (spec A1 step 5).

    Callable identically from a workflow task and from a notebook cell::

        from src.pipelines import fit_models
        fit_models.main(spark, config)

    Exactly one ``bronze.ingestion_runs`` row is written per call, on success and on failure, via
    :func:`write_gold`.
    """
    catalog = catalog or str(config["catalog"])
    require_table(spark, qualified(catalog, FEATURES_TABLE))

    universe = (
        [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        if tickers is not None
        else tickers_with_features(spark, catalog)
    )
    if not universe:
        log.warning("%s: no tickers to fit — has build_features run?", TASK_NAME)

    summary = FitSummary()
    generated_at = utc_now()  # one timestamp for the whole run, shared by every forecast row

    for ticker in universe:
        fit = fit_ticker(ticker, read_features(spark, catalog, ticker), config)
        summary.fits.append(fit)
        if not fit.ok:
            log.warning("%s %s skipped: %s", TASK_NAME, ticker, fit.skipped_reason)
            summary.skipped.append({"ticker": ticker, "reason": fit.skipped_reason})
            continue
        fit.forecast_row["generated_at"] = generated_at
        log.info(
            "%s %s as_of=%s model_used=%s return_p50=%+.4f prob_positive=%.3f",
            TASK_NAME,
            ticker,
            fit.as_of_date,
            fit.model_used,
            fit.forecast_row["return_p50"],
            fit.forecast_row["prob_positive"],
        )

    writes: dict[str, list[dict]] = {}
    regime_rows = [fit.regime_row for fit in summary.fits if fit.regime_row is not None]
    forecast_rows = [fit.forecast_row for fit in summary.fits if fit.forecast_row is not None]
    if regime_rows:
        writes[GOLD_REGIME_STATES] = regime_rows
    if forecast_rows:
        writes[GOLD_FORECAST_RUNS] = forecast_rows

    # Called even with nothing to write: the ledger's contract is one row per task per run, and a
    # run where every ticker failed is precisely the run an operator needs to find in the ledger.
    written = write_gold(spark, catalog, TASK_NAME, writes)
    summary.run_id = written["run_id"]
    summary.rows_written = written["rows_written"]
    return summary.as_dict()
