"""Databricks Workflow definition (spec C-6). Idempotent: create it once, re-run to update it.

THE DAILY JOB, in A1 order, one task per step, every task the same thin dispatcher notebook
(``notebooks/20_daily_run.py``) with a different ``task`` parameter:

    ingest_prices -> ingest_news -> build_silver -> build_features -> refresh_news_recent
    -> fit_models -> sync_news_index -> sync_lakebase_history -> [backfill_news_recent]

Run it::

    python setup/create_workflow.py            # create or update, then print the task graph
    python setup/create_workflow.py --dry-run  # print the graph, touch nothing

WHY A STRICT CHAIN. Several of these could run in parallel — ``ingest_news`` needs nothing from
``ingest_prices``, and ``refresh_news_recent``'s real dependency is ``build_silver`` rather than
``build_features``. They are chained anyway. The two ingestion tasks share one rate-limited vendor
API (5 requests/minute, config ``massive.rate_limit_per_min``) and running them together halves the
budget each one thinks it has; the rest are minutes of work on tiny tables, where a serial run
history that reads top to bottom is worth more than the wall clock it costs.

RETRIES ON THE INGESTION TASKS ONLY (spec C-6). Those fail for transient network reasons and a
retry fixes them. A failed MLE fit fails identically the second time, and retrying it just delays
the failure an operator needs to see.

``backfill_news_recent`` is included only when ``news_recent.include_backfill`` is true — the same
key ``refresh_news_recent`` reads to decide whether to stop deleting the history the backfill
builds. One key, so the two halves cannot disagree.

DEPENDENCIES COME FROM THE NOTEBOOK'S %pip CELL, not from a job environment. ``JobEnvironment`` /
``compute.Environment`` is documented in the installed SDK as the environment for NON-NOTEBOOK
tasks; these are notebook tasks, so ``%pip install -r ../requirements-databricks.txt`` at the top
of the dispatcher is what puts statsmodels and exchange_calendars on the cluster (spec C-c). No
compute is declared at all, which is what makes these serverless tasks.

The walk-forward backtest is deliberately NOT in this workflow — it is a separate on-demand job
(``notebooks/10_backtest_run.py``), because a verdict about the models should be produced when
someone is looking at it.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "TASKS",
    "WorkflowSettings",
    "WorkflowTask",
    "build_tasks",
    "find_job",
    "job_settings",
    "main",
    "task_keys",
    "workflow_settings",
]

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

#: The optional task. Present in the job only when ``news_recent.include_backfill`` is true.
BACKFILL_TASK = "backfill_news_recent"


@dataclass(frozen=True)
class WorkflowTask:
    """One task: what it is called, what it waits for, and how often it may retry."""

    key: str
    depends_on: tuple[str, ...]
    description: str
    #: Whether transient failure is plausible. The COUNT lives in config
    #: (``workflow.ingestion_retries``); this only says which tasks may use it.
    retryable: bool = False
    optional: bool = False


#: The graph, declared once. ``tests/test_workflow.py`` asserts this against
#: ``notebooks/20_daily_run.py``'s dispatch table, so a task the job schedules and the notebook
#: cannot run is caught here rather than at 22:30 UTC.
TASKS: tuple[WorkflowTask, ...] = (
    WorkflowTask(
        key="ingest_prices",
        depends_on=(),
        retryable=True,
        description="Massive daily bars per ticker since the stored watermark -> bronze.prices_raw.",
    ),
    WorkflowTask(
        key="ingest_news",
        depends_on=("ingest_prices",),
        retryable=True,
        description="Massive news per ticker since the stored watermark -> bronze.news_raw.",
    ),
    WorkflowTask(
        key="build_silver",
        depends_on=("ingest_news",),
        description="bronze -> silver.daily_prices and silver.news_articles, in one task.",
    ),
    WorkflowTask(
        key="build_features",
        depends_on=("build_silver",),
        description="silver -> silver.daily_features, the contract with the modeling layer.",
    ),
    WorkflowTask(
        key="refresh_news_recent",
        depends_on=("build_features",),
        description="Republish the rolling news window the app reads; trim what fell out of it.",
    ),
    WorkflowTask(
        key="fit_models",
        depends_on=("refresh_news_recent",),
        description="Per ticker: full-history ladder fit -> gold.regime_states + gold.forecast_runs.",
    ),
    WorkflowTask(
        key="sync_news_index",
        depends_on=("fit_models",),
        description="Trigger the AI Search Delta Sync index over silver.news_articles.",
    ),
    WorkflowTask(
        key="sync_lakebase_history",
        depends_on=("sync_news_index",),
        description="Watermark CDC: Lakebase watchlist and reports -> gold.lb_*_history.",
    ),
    WorkflowTask(
        key=BACKFILL_TASK,
        depends_on=("sync_lakebase_history",),
        description="Optional: extend the news_recent window one batch further back.",
        optional=True,
    ),
)


@dataclass(frozen=True)
class WorkflowSettings:
    """The ``workflow`` config block, plus the one key it borrows from ``news_recent``."""

    job_name: str
    notebook_path: str
    schedule_quartz: str
    timezone: str
    paused: bool
    ingestion_retries: int
    include_backfill: bool


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def workflow_settings(config: Mapping[str, Any] | None = None) -> WorkflowSettings:
    """Read the ``workflow`` block, refusing a placeholder notebook path.

    The path is the one value that cannot have a sensible default: it is where the Git folder
    landed in a particular workspace. Failing here beats a job whose every task reports
    "notebook not found" at 22:30 UTC.
    """
    source = config if config is not None else load_config()
    section = dict(source.get("workflow") or {})
    news = dict(source.get("news_recent") or {})

    notebook_path = str(section.get("notebook_path") or "").strip()
    if not notebook_path or "REPLACE_ME" in notebook_path:
        raise ValueError(
            "workflow.notebook_path is not set. Point it at notebooks/20_daily_run in the target "
            "workspace, e.g. /Workspace/Repos/you@example.com/regime-market-agent/notebooks/"
            "20_daily_run (no .py suffix)."
        )

    return WorkflowSettings(
        job_name=str(section.get("job_name") or "regime-market-daily"),
        notebook_path=notebook_path,
        schedule_quartz=str(section.get("schedule_quartz") or "0 30 22 * * ?"),
        timezone=str(section.get("timezone") or "UTC"),
        paused=bool(section.get("paused", True)),
        ingestion_retries=int(section.get("ingestion_retries", 2)),
        include_backfill=bool(news.get("include_backfill", False)),
    )


def task_keys(settings: WorkflowSettings) -> list[str]:
    """The tasks this configuration actually schedules, in declaration order."""
    return [task.key for task in TASKS if settings.include_backfill or not task.optional]


def build_tasks(settings: WorkflowSettings) -> list[Any]:
    """The SDK ``Task`` objects. Every task is the same notebook with a different parameter."""
    from databricks.sdk.service.jobs import NotebookTask, Source, Task, TaskDependency

    scheduled = set(task_keys(settings))
    tasks: list[Any] = []

    for task in TASKS:
        if task.key not in scheduled:
            continue
        retries = settings.ingestion_retries if task.retryable else 0
        tasks.append(
            Task(
                task_key=task.key,
                description=task.description,
                notebook_task=NotebookTask(
                    notebook_path=settings.notebook_path,
                    base_parameters={"task": task.key},
                    source=Source.WORKSPACE,
                ),
                depends_on=[TaskDependency(task_key=key) for key in task.depends_on] or None,
                max_retries=retries or None,
            )
        )
    return tasks


def job_settings(settings: WorkflowSettings) -> dict:
    """The job definition as keyword arguments, shared by ``jobs.create`` and ``jobs.reset``.

    ``max_concurrent_runs=1``: every write in this pipeline is a MERGE, and two runs merging the
    same day's rows concurrently is the one way to make an idempotent write non-deterministic.
    """
    from databricks.sdk.service.jobs import CronSchedule, PauseStatus, QueueSettings

    return {
        "name": settings.job_name,
        "tasks": build_tasks(settings),
        "schedule": CronSchedule(
            quartz_cron_expression=settings.schedule_quartz,
            timezone_id=settings.timezone,
            pause_status=PauseStatus.PAUSED if settings.paused else PauseStatus.UNPAUSED,
        ),
        "max_concurrent_runs": 1,
        "queue": QueueSettings(enabled=True),
    }


def find_job(w: Any, name: str) -> Any | None:
    """The existing job with this exact name, or ``None``.

    Looked up by name rather than by a stored job id because there is nowhere in this repo to
    store an id that would not itself need syncing. Two jobs sharing a name is a workspace someone
    has already broken by hand, and it is reported rather than guessed at.
    """
    matches = [job for job in w.jobs.list(name=name) if getattr(job, "settings", None)]
    matches = [job for job in matches if job.settings.name == name]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} jobs are named {name!r}; delete the duplicates before re-running "
            "this script, which cannot know which one is yours."
        )
    return matches[0]


def main(
    w: Any = None,
    config: Mapping[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Create the job, or overwrite the existing one with this definition. Safe to re-run.

    ``reset`` rather than ``update``: ``update`` merges task arrays by ``task_key``, so a task
    removed from :data:`TASKS` — turning the optional backfill back off, say — would survive in the
    workspace. Overwriting means the file is the definition.
    """
    settings = workflow_settings(config)
    plan = {
        "job_name": settings.job_name,
        "notebook_path": settings.notebook_path,
        "schedule": f"{settings.schedule_quartz} {settings.timezone}",
        "paused": settings.paused,
        "ingestion_retries": settings.ingestion_retries,
        "tasks": task_keys(settings),
    }

    if dry_run:
        log.info("dry run: %s", plan)
        return {"created": False, "updated": False, **plan}

    if w is None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

    definition = job_settings(settings)
    existing = find_job(w, settings.job_name)

    if existing is None:
        response = w.jobs.create(**definition)
        job_id = getattr(response, "job_id", None)
        log.info("created job %s id=%s", settings.job_name, job_id)
        return {"created": True, "updated": False, "job_id": job_id, **plan}

    from databricks.sdk.service.jobs import JobSettings

    job_id = existing.job_id
    w.jobs.reset(job_id=job_id, new_settings=JobSettings(**definition))
    log.info("updated job %s id=%s", settings.job_name, job_id)
    return {"created": False, "updated": True, "job_id": job_id, **plan}


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create or update the daily workflow (spec C-6).")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the task graph without touching the workspace",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = main(config=load_config(args.config), dry_run=args.dry_run)

    print(f"job: {result['job_name']}  schedule: {result['schedule']}  paused: {result['paused']}")
    print(f"notebook: {result['notebook_path']}")
    for task in TASKS:
        if task.key not in result["tasks"]:
            continue
        after = ", ".join(task.depends_on) or "-"
        retries = result["ingestion_retries"] if task.retryable else 0
        print(f"  {task.key:<24} after={after:<24} retries={retries}")


if __name__ == "__main__":
    _cli()
