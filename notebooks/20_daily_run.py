# Databricks notebook source
# MAGIC %pip install -r ../requirements-databricks.txt

# COMMAND ----------

"""The daily workflow's entry point (spec C-6). A THIN WRAPPER — no logic lives here.

ONE NOTEBOOK, ONE ``task`` PARAMETER, NINE TASKS. Every task in the job runs this file with a
different ``task`` value. The alternative — a ``spark_python_task`` per module — puts the repo root
on ``sys.path`` nowhere, and ``src.*`` imports then resolve only by accident of the working
directory. This file does that once, in :func:`ensure_repo_on_path`, the same way the backtest
notebook does (spec C-a), and every task inherits it.

    In a workspace : set the ``task`` widget, or let the job's base_parameters set it.
    On the CLI     : python notebooks/20_daily_run.py --task build_features

The task order and the dependencies between tasks are declared in ``setup/create_workflow.py``,
not here. This file knows how to run ONE task and nothing about what runs before or after it —
which is what makes a single failed task re-runnable on its own from the Jobs UI.
"""

import logging
import sys
from pathlib import Path

log = logging.getLogger("daily_run")


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


# The path has to be set before the imports below, which is why they are not at the top.
ROOT = ensure_repo_on_path()

from setup.create_ai_search import search_settings, trigger_sync  # noqa: E402 — see above
from src.ingestion import ingest_news, ingest_prices  # noqa: E402
from src.pipelines import (  # noqa: E402
    feature_pipeline,
    fit_models,
    lakebase_history,
    news_recent,
    silver_news,
    silver_prices,
)


def build_silver(spark, config: dict) -> dict:
    """Both silver builds, in one task (spec C-6).

    They are one task because they are one dependency: ``build_features`` reads prices AND news,
    so splitting them would only add a fan-in with no independent consumer of either half.
    """
    return {
        "prices": silver_prices.main(spark, config),
        "news": silver_news.main(spark, config),
    }


def sync_news_index(spark, config: dict) -> dict:
    """Trigger the AI Search index sync (C-1). The only task that does not touch Spark."""
    return {"index": trigger_sync(settings=search_settings(config))}


#: task name -> callable(spark, config). The names are the job's task keys, and
#: tests/test_workflow.py asserts the two sets are identical — a task the job schedules but this
#: file cannot run would fail at 22:30 UTC rather than at edit time.
TASKS = {
    "ingest_prices": lambda spark, config: ingest_prices.main(spark, config),
    "ingest_news": lambda spark, config: ingest_news.main(spark, config),
    "build_silver": build_silver,
    "build_features": lambda spark, config: feature_pipeline.main(spark, config),
    "refresh_news_recent": lambda spark, config: news_recent.refresh(spark, config),
    "fit_models": lambda spark, config: fit_models.main(spark, config),
    "sync_news_index": sync_news_index,
    "sync_lakebase_history": lambda spark, config: lakebase_history.main(spark, config),
    "backfill_news_recent": lambda spark, config: news_recent.backfill(spark, config),
}


def notebook_dbutils():
    """The ``dbutils`` a workspace injects into notebook globals, or ``None`` locally."""
    try:
        return dbutils  # type: ignore[name-defined]  # noqa: F821 — injected by the runtime
    except NameError:
        return None


def read_parameters(argv=None) -> dict:
    """The task name, from a widget in a workspace and from a flag on the CLI."""
    dbu = notebook_dbutils()
    if dbu is not None:
        dbu.widgets.text("task", "", "Task (see TASKS)")
        return {"task": dbu.widgets.get("task"), "config": None}

    import argparse

    parser = argparse.ArgumentParser(description="One daily workflow task (spec C-6).")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    return {"task": args.task, "config": args.config}


def load_config(root: Path, path: str | None = None) -> dict:
    import yaml

    with open(path or root / "config" / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(spark, config: dict, task: str) -> dict:
    """Run one task by name. Wiring only."""
    name = str(task).strip()
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; expected one of {', '.join(sorted(TASKS))}")

    print(f"task={name} catalog={config['catalog']}")
    result = TASKS[name](spark, config)
    print(f"task={name} done: {result}")
    return result


def _cli() -> None:
    """Wiring only: parameters, config, session, :func:`main`."""
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO)
    params = read_parameters()
    config = load_config(ROOT, params["config"])
    main(SparkSession.builder.getOrCreate(), config, params["task"])


if __name__ == "__main__":
    _cli()
