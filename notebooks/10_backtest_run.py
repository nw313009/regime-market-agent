# Databricks notebook source
# MAGIC %pip install -r ../requirements-databricks.txt

# COMMAND ----------

"""B-5/B-6 walk-forward backtest runner. A THIN WRAPPER (spec C-a) — no logic lives here.

What it does: read ``silver.daily_features`` per ticker through one ``.toPandas()``, run the
walk-forward backtest from ``src/models/backtest.py``, print the pooled table and the per-model
fallback rates, and MERGE the results into gold through ``src/pipelines/write_gold``.

RUN A TWO-TICKER VALIDATION FIRST. The defaults are deliberately ``NVDA,MSFT`` rather than the
full seed universe: the full run is 5 tickers x 26 origins x 3 arms = 390 MLE fits, and finding out
after all of them that a ticker's features were never built is an expensive way to learn it. Widen
to ``all`` once a two-ticker run comes back with a sane pooled table.

    In a workspace : set the ``tickers`` and ``n_weeks`` widgets at the top of the notebook.
    On the CLI     : python notebooks/10_backtest_run.py --tickers NVDA,MSFT --n-weeks 4

Both knobs override config; a blank ``n_weeks`` means "use ``config/config.yaml``". The config file
on disk is never modified — the override is applied to a copy, so the file stays the record of what
the project ships.

READ THE FALLBACK RATE BEFORE THE BRIER SCORES. A Model C column with a 40% fallback rate is not a
Model C result: 40% of those rows were produced by Model B or A after a failed fit, which is exactly
what ``model_used`` records. The breakdown printed below reports it per arm, and ``n`` is printed
next to every score because 26 weekly origins on two tickers is 52 forecasts per model — a small
sample for a Brier difference, and "no meaningful improvement detected at this sample size" is a
legitimate finding.

WHAT REACHES gold, AND WHAT DOES NOT:

- ``gold.backtest_metrics`` — one row per (origin, ticker, model). The whole point of the run.
- ``gold.backtest_summary`` — the pooled row per model, replacing the previous run's.
- ``gold.forecast_runs``    — ONE row per ticker: the production ladder's forecast at the most
  recent origin. The per-origin backtest forecasts are deliberately NOT dumped here.
  ``forecast_runs`` is keyed ``(ticker, as_of_date, model_used)`` and means "the current forecast";
  writing 78 historical per-arm forecasts into it would collide on that key the moment arm C fell
  back to Markov at an origin where arm B also used Markov. Their scores live in
  ``backtest_metrics``, which is keyed on the arm and cannot collide.
- ``gold.regime_states``    — not written here. It needs the sorted regime parameters rather than a
  forecast summary, and it belongs to the daily ``fit_models`` task.

The forecast is taken at the last BACKTEST ORIGIN, not at the last session, because an origin is by
definition a window this checkpoint's tested code can build. Forecasting from the final session is
the daily ``run_forecasts`` task's job and lands with it.

Everything is a MERGE on declared keys, so re-running this notebook is safe (rule 4). Every run
leaves exactly one ``bronze.ingestion_runs`` row, on success and on failure.

Prerequisites: ``setup/create_catalog.sql``, then ``setup/create_delta_tables.sql``, then the A-2 to
A-4 tasks, so ``silver.daily_features`` is populated. A ticker with no rows is reported and skipped,
never silently scored as zero origins.
"""

import logging
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger("backtest_run")

#: Two tickers, not five. See the docstring: validate cheaply, then widen.
DEFAULT_TICKERS = "NVDA,MSFT"


def repo_root() -> Path:
    """The repo root, whether this runs as a file or is pasted into a notebook cell."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # notebook cell: no __file__
        cwd = Path.cwd()
        return cwd.parent if cwd.name == "notebooks" else cwd


def ensure_repo_on_path() -> Path:
    """Put the repo root on ``sys.path`` so ``src.*`` imports resolve (spec C-a)."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# The path has to be set before the src imports below, which is why they are not at the top of the
# file. This is the standard notebook wrapper shape; tests/conftest.py does the same thing.
ROOT = ensure_repo_on_path()

from src.models.backtest import (  # noqa: E402 — see the comment above
    FEATURE_COLUMNS,
    MODEL_ARMS,
    TASK_NAME,
    fit_arm,
    origin_window,
    run_backtest,
    weekly_origins,
)
from src.models.monte_carlo import forecast_config  # noqa: E402
from src.pipelines import (  # noqa: E402
    GOLD_BACKTEST_METRICS,
    GOLD_BACKTEST_SUMMARY,
    GOLD_FORECAST_RUNS,
    forecast_run_row,
    utc_now,
    write_gold,
)

#: The production ladder is the richest arm's ladder: C, falling back to B then A.
PRODUCTION_ARM = MODEL_ARMS[0]


# ------------------------------------------------------------------ parameters


def notebook_dbutils():
    """The ``dbutils`` a workspace injects into notebook globals, or ``None`` locally."""
    try:
        return dbutils  # type: ignore[name-defined]  # noqa: F821 — injected by the runtime
    except NameError:
        return None


def read_parameters(argv=None) -> dict:
    """The same two knobs from widgets in a workspace and from flags on the CLI."""
    dbu = notebook_dbutils()
    if dbu is not None:
        dbu.widgets.text("tickers", DEFAULT_TICKERS, "Tickers (comma-separated, or 'all')")
        dbu.widgets.text("n_weeks", "", "Origins per ticker (blank = config)")
        return {
            "tickers": dbu.widgets.get("tickers"),
            "n_weeks": dbu.widgets.get("n_weeks"),
            "config": None,
        }

    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtest (spec B-5).")
    parser.add_argument("--tickers", default=DEFAULT_TICKERS, help="comma-separated, or 'all'")
    parser.add_argument("--n-weeks", default="", help="origins per ticker; blank = config")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    return {"tickers": args.tickers, "n_weeks": args.n_weeks, "config": args.config}


def load_config(root: Path, path: str | None = None) -> dict:
    import yaml

    with open(path or root / "config" / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict, n_weeks: str) -> dict:
    """A COPY of the config with ``n_weeks`` overridden; the file on disk stays the record."""
    import copy

    resolved = copy.deepcopy(config)
    if str(n_weeks).strip():
        resolved["backtest"]["n_weeks"] = int(n_weeks)
    return resolved


def resolve_tickers(config: dict, requested: str) -> list[str]:
    """The requested subset, or the whole seed universe for ``all`` / a blank value."""
    seed = sorted({str(t).strip().upper() for t in (config.get("tickers") or {}).get("seed") or []})
    if requested.strip().lower() in ("", "all"):
        return seed

    wanted = sorted({t.strip().upper() for t in requested.split(",") if t.strip()})
    unknown = [ticker for ticker in wanted if ticker not in seed]
    if unknown:
        # Not fatal: a watchlist ticker added through the agent is legitimately outside the seed.
        # Printed output stays ASCII: a Windows console is cp1252 and mangles anything else.
        print(f"note: {', '.join(unknown)} not in config tickers.seed (expected for watchlist adds)")
    return wanted


# ------------------------------------------------------------------ the Spark boundary


def read_features(spark, catalog: str, ticker: str):
    """One ticker of ``silver.daily_features`` as a pandas frame (spec C-b).

    THE ONLY Spark call in the backtest path. Ordered by ``trade_date`` because the backtest slices
    training windows positionally, and projected to the columns the models read so the driver is not
    handed columns nobody uses. The ticker travels as a parameter marker rather than in the SQL text.
    """
    columns = ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
    return spark.sql(
        f"SELECT {columns} FROM {catalog}.silver.daily_features "
        "WHERE ticker = :ticker ORDER BY trade_date",
        args={"ticker": ticker},
    ).toPandas()


def latest_forecast_row(ticker: str, frame, config: dict) -> dict | None:
    """The production ladder's forecast at the most recent origin, as a ``forecast_runs`` row.

    One extra fit per ticker, deliberately: it exercises the fit -> forecast -> gold path the daily
    job will use, which is the half of B-6 the backtest metrics do not cover.
    """
    min_train_days = int(config["backtest"]["min_train_days"])
    horizon_days = forecast_config(config).horizon_days

    origins = weekly_origins(
        frame, n_weeks=1, min_train_days=min_train_days, horizon_days=horizon_days
    )
    if not origins:
        return None

    window = origin_window(frame, origins[-1], horizon_days=horizon_days)
    fit = fit_arm(PRODUCTION_ARM, window, config, min_obs=min_train_days)
    if fit is None:
        print(f"  {ticker}: every rung failed at {window.origin}; no forecast row")
        return None

    print(
        f"  {ticker}: forecast as_of={window.origin} model_used={fit.model_used} "
        f"return_p50={fit.summary.return_p50:+.4f} prob_positive={fit.summary.prob_positive:.3f}"
    )
    return forecast_run_row(fit.summary, ticker=ticker, as_of_date=window.origin)


# ------------------------------------------------------------------ output


def print_pooled(result) -> None:
    """The pooled table. ``n`` is printed next to every score, never as a footnote."""
    header = (
        f"{'model':<13}{'n':>6}{'tickers':>9}{'brier':>9}{'mae':>9}{'cover80':>9}{'fallback':>10}"
    )
    print("\npooled across tickers (spec B-5)")
    print(header)
    print("-" * len(header))
    for row in result.pooled:
        print(
            f"{row.model:<13}{row.n:>6}{row.n_tickers:>9}{row.brier:>9.4f}"
            f"{row.mae:>9.4f}{row.coverage_80:>9.3f}{row.fallback_rate:>10.3f}"
        )
    print("\ncoverage_80 should sit near 0.800 from either side; brier 0.25 is a coin flip.")


def print_fallbacks(result) -> None:
    """Which rung actually answered, per arm. The number to read before any Brier comparison."""
    print("\nfallback breakdown: model asked for -> model_used")
    for arm in MODEL_ARMS:
        used = Counter(row.model_used for row in result.rows if row.model == arm)
        if not used:
            continue
        detail = ", ".join(f"{name}={count}" for name, count in sorted(used.items()))
        print(f"  {arm:<13} {detail}")

    # Truncated verbatim rather than parsed: the reason is already prefixed with the rung that
    # refused, and splitting on the colons in it would only find that prefix again.
    reasons = Counter(row.failure_reason[:80] for row in result.rows if row.failure_reason)
    for reason, count in reasons.most_common(5):
        print(f"    x{count} {reason}")

    if result.skipped:
        print(f"\n{len(result.skipped)} (origin, model) pairs produced no forecast at all:")
        for entry in result.skipped[:10]:
            print(f"  {entry['ticker']} {entry['origin_date']} {entry['model']}")


# ------------------------------------------------------------------ the run


def main(spark, config: dict, tickers: list[str], catalog: str | None = None) -> dict:
    """Read, backtest, report, write. Wiring only — every computation lives in ``src/``."""
    catalog = catalog or config["catalog"]
    print(
        f"backtest catalog={catalog} tickers={','.join(tickers)} "
        f"n_weeks={config['backtest']['n_weeks']} "
        f"min_train_days={config['backtest']['min_train_days']}"
    )

    frames = {}
    for ticker in tickers:
        frame = read_features(spark, catalog, ticker)
        if frame.empty:
            print(f"  {ticker}: 0 sessions. Run the A-2..A-4 tasks first. Skipped.")
            continue
        print(f"  {ticker}: {len(frame)} sessions, {frame['trade_date'].iloc[0]} .. {frame['trade_date'].iloc[-1]}")
        frames[ticker] = frame

    if not frames:
        raise SystemExit("no features for any requested ticker — nothing to backtest")

    result = run_backtest(frames, config)
    if not result.rows:
        raise SystemExit(
            "the backtest produced no scored rows — check min_train_days against the history "
            "each ticker actually has"
        )

    print_pooled(result)
    print_fallbacks(result)

    print("\nlatest ladder forecast per ticker")
    forecast_rows = [
        row
        for ticker, frame in frames.items()
        if (row := latest_forecast_row(ticker, frame, config)) is not None
    ]

    computed_at = utc_now()  # one timestamp for the whole run, shared by every pooled row
    writes = {
        GOLD_BACKTEST_METRICS: [row.as_row() for row in result.rows],
        GOLD_BACKTEST_SUMMARY: [row.as_row(computed_at) for row in result.pooled],
    }
    if forecast_rows:
        writes[GOLD_FORECAST_RUNS] = forecast_rows

    summary = write_gold(spark, catalog, TASK_NAME, writes)

    print(f"\nwrote gold run_id={summary['run_id']} rows={summary['rows_total']}")
    for table, count in summary["rows_written"].items():
        print(f"  {table:<28} {count}")
    return summary


def _cli() -> None:
    """Wiring only: parameters, config, session, :func:`main`."""
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO)

    params = read_parameters()
    config = apply_overrides(load_config(ROOT, params["config"]), params["n_weeks"])
    tickers = resolve_tickers(config, params["tickers"])

    main(SparkSession.builder.getOrCreate(), config, tickers)


if __name__ == "__main__":
    _cli()
