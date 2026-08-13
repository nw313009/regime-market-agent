"""Daily workflow tests (spec C-6).

The job is a declaration, and a declaration's failure mode is that it is wrong in a way nothing
notices until 22:30 UTC. These tests pin the parts that would fail then:

- THE TASK LIST AND ITS EDGES. The A1 order, as a chain, with the optional backfill in or out.
- THE DISPATCHER AGREES. Every task the job schedules is a task ``notebooks/20_daily_run.py`` can
  actually run, and vice versa. Two lists of task names in two files is exactly the kind of thing
  that drifts by one rename.
- RETRIES ON THE INGESTION TASKS ONLY, with the count coming from config.
- CREATE VERSUS UPDATE. Re-running the script must not produce a second job, and updating must
  REPLACE the task list rather than merge into it — otherwise a task removed here survives in the
  workspace forever.

The notebook is read as a module rather than executed: importing it would import pyspark-free but
workspace-shaped modules and, more to the point, the dispatch table is the thing under test, not
the import.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from setup import create_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "20_daily_run.py"

#: The order spec C-6 declares, including the optional last task.
EXPECTED_ORDER = [
    "ingest_prices",
    "ingest_news",
    "build_silver",
    "build_features",
    "refresh_news_recent",
    "fit_models",
    "sync_news_index",
    "sync_lakebase_history",
    "backfill_news_recent",
]


@pytest.fixture(scope="module")
def dispatcher():
    """``notebooks/20_daily_run.py`` as a module. It is a script, so it is loaded by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("daily_run_notebook", NOTEBOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config() -> dict:
    """The shipped config with a usable notebook path — the file ships a REPLACE_ME on purpose."""
    resolved = copy.deepcopy(yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text("utf-8")))
    resolved["workflow"]["notebook_path"] = "/Workspace/Repos/demo/regime-market-agent/notebooks/20_daily_run"
    return resolved


@pytest.fixture
def settings(config):
    return create_workflow.workflow_settings(config)


class FakeJobsAPI:
    def __init__(self, existing: list | None = None):
        self.existing = list(existing or [])
        self.created: list[dict] = []
        self.reset_calls: list[tuple[int, object]] = []

    def list(self, name: str | None = None):
        return iter(self.existing)

    def create(self, **kwargs):
        self.created.append(kwargs)
        return type("CreateResponse", (), {"job_id": 4242})()

    def reset(self, job_id: int, new_settings):
        self.reset_calls.append((job_id, new_settings))


class FakeWorkspace:
    def __init__(self, existing: list | None = None):
        self.jobs = FakeJobsAPI(existing)


def existing_job(name: str, job_id: int = 99):
    settings = type("Settings", (), {"name": name})()
    return type("Job", (), {"job_id": job_id, "settings": settings})()


class TestTaskGraph:
    def test_the_tasks_are_the_a1_order(self):
        assert [task.key for task in create_workflow.TASKS] == EXPECTED_ORDER

    def test_every_task_depends_on_the_one_before_it(self):
        """A strict chain: two ingestion tasks in parallel would halve the vendor rate budget."""
        for previous, task in zip(create_workflow.TASKS, create_workflow.TASKS[1:]):
            assert task.depends_on == (previous.key,), f"{task.key} should follow {previous.key}"
        assert create_workflow.TASKS[0].depends_on == ()

    def test_the_backfill_is_the_only_optional_task(self):
        optional = [task.key for task in create_workflow.TASKS if task.optional]
        assert optional == ["backfill_news_recent"]

    def test_the_backfill_is_excluded_by_default(self, settings):
        assert "backfill_news_recent" not in create_workflow.task_keys(settings)
        assert create_workflow.task_keys(settings) == EXPECTED_ORDER[:-1]

    def test_one_config_key_turns_the_backfill_on(self, config):
        """The same key ``refresh_news_recent`` reads for its retention floor (see news_recent)."""
        config["news_recent"]["include_backfill"] = True
        settings = create_workflow.workflow_settings(config)

        assert create_workflow.task_keys(settings) == EXPECTED_ORDER

    def test_only_the_ingestion_tasks_retry(self):
        retryable = [task.key for task in create_workflow.TASKS if task.retryable]
        assert retryable == ["ingest_prices", "ingest_news"]

    def test_the_dispatcher_runs_exactly_the_tasks_the_job_schedules(self, dispatcher):
        """Two lists of task names in two files; this is what keeps them one list.

        The notebook is IMPORTED rather than parsed, so this also proves its imports resolve —
        a task that fails at ``import src.pipelines.fit_models`` fails at 22:30 UTC otherwise.
        """
        assert set(dispatcher.TASKS) == {task.key for task in create_workflow.TASKS}

    def test_the_dispatcher_refuses_an_unknown_task(self, dispatcher):
        with pytest.raises(SystemExit, match="unknown task"):
            dispatcher.main(spark=None, config={"catalog": "market_intel"}, task="not_a_task")


class TestSettings:
    def test_a_placeholder_notebook_path_is_refused(self, config):
        config["workflow"]["notebook_path"] = "/Workspace/Repos/REPLACE_ME/x/notebooks/20_daily_run"
        with pytest.raises(ValueError, match="notebook_path"):
            create_workflow.workflow_settings(config)

    def test_an_empty_notebook_path_is_refused(self, config):
        config["workflow"]["notebook_path"] = ""
        with pytest.raises(ValueError, match="notebook_path"):
            create_workflow.workflow_settings(config)

    def test_the_shipped_defaults_are_a_daily_utc_schedule(self, settings):
        assert settings.schedule_quartz == "0 30 22 * * ?"
        assert settings.timezone == "UTC"
        assert settings.ingestion_retries == 2

    def test_the_job_ships_paused(self, settings):
        """A job that schedules itself the moment it is defined fires against a half-built workspace."""
        assert settings.paused is True


class TestJobDefinition:
    def test_every_task_is_the_same_notebook_with_its_own_parameter(self, settings):
        tasks = create_workflow.build_tasks(settings)

        assert [task.task_key for task in tasks] == EXPECTED_ORDER[:-1]
        for task in tasks:
            assert task.notebook_task.notebook_path == settings.notebook_path
            assert task.notebook_task.base_parameters == {"task": task.task_key}

    def test_the_dependencies_survive_into_the_sdk_objects(self, settings):
        tasks = {task.task_key: task for task in create_workflow.build_tasks(settings)}

        assert tasks["ingest_prices"].depends_on is None
        assert [dep.task_key for dep in tasks["fit_models"].depends_on] == ["refresh_news_recent"]
        assert [dep.task_key for dep in tasks["build_silver"].depends_on] == ["ingest_news"]

    def test_retries_are_set_from_config_on_the_ingestion_tasks_only(self, settings):
        tasks = {task.task_key: task for task in create_workflow.build_tasks(settings)}

        assert tasks["ingest_prices"].max_retries == 2
        assert tasks["ingest_news"].max_retries == 2
        assert tasks["fit_models"].max_retries is None
        assert tasks["sync_lakebase_history"].max_retries is None

    def test_the_schedule_is_declared_paused(self, settings):
        from databricks.sdk.service.jobs import PauseStatus

        definition = create_workflow.job_settings(settings)

        assert definition["schedule"].quartz_cron_expression == "0 30 22 * * ?"
        assert definition["schedule"].timezone_id == "UTC"
        assert definition["schedule"].pause_status == PauseStatus.PAUSED

    def test_unpausing_is_a_config_change(self, config):
        from databricks.sdk.service.jobs import PauseStatus

        config["workflow"]["paused"] = False
        definition = create_workflow.job_settings(create_workflow.workflow_settings(config))

        assert definition["schedule"].pause_status == PauseStatus.UNPAUSED

    def test_only_one_run_of_this_job_may_be_in_flight(self, settings):
        """Two runs MERGEing the same day is the one way to make an idempotent write ambiguous."""
        assert create_workflow.job_settings(settings)["max_concurrent_runs"] == 1

    def test_no_compute_is_declared_so_the_tasks_are_serverless(self, settings):
        for task in create_workflow.build_tasks(settings):
            assert getattr(task, "new_cluster", None) is None
            assert getattr(task, "existing_cluster_id", None) is None
            assert getattr(task, "job_cluster_key", None) is None


class TestIdempotency:
    def test_a_missing_job_is_created(self, config):
        w = FakeWorkspace()

        result = create_workflow.main(w, config)

        assert result["created"] is True and result["job_id"] == 4242
        assert len(w.jobs.created) == 1
        assert w.jobs.created[0]["name"] == "regime-market-daily"

    def test_an_existing_job_is_overwritten_rather_than_duplicated(self, config):
        w = FakeWorkspace([existing_job("regime-market-daily")])

        result = create_workflow.main(w, config)

        assert result["updated"] is True and result["job_id"] == 99
        assert w.jobs.created == []
        job_id, new_settings = w.jobs.reset_calls[0]
        assert job_id == 99
        assert [task.task_key for task in new_settings.tasks] == EXPECTED_ORDER[:-1]

    def test_a_job_that_only_shares_a_prefix_is_not_ours(self, config):
        """``jobs.list(name=...)`` filters server-side; this is the belt to that braces."""
        w = FakeWorkspace([existing_job("regime-market-daily-old")])

        result = create_workflow.main(w, config)

        assert result["created"] is True

    def test_duplicate_jobs_are_reported_rather_than_guessed_at(self, config):
        w = FakeWorkspace([existing_job("regime-market-daily", 1), existing_job("regime-market-daily", 2)])

        with pytest.raises(RuntimeError, match="delete the duplicates"):
            create_workflow.main(w, config)

    def test_a_dry_run_touches_nothing(self, config):
        w = FakeWorkspace()

        result = create_workflow.main(w, config, dry_run=True)

        assert result == {
            "created": False,
            "updated": False,
            "job_name": "regime-market-daily",
            "notebook_path": "/Workspace/Repos/demo/regime-market-agent/notebooks/20_daily_run",
            "schedule": "0 30 22 * * ? UTC",
            "paused": True,
            "ingestion_retries": 2,
            "tasks": EXPECTED_ORDER[:-1],
        }
        assert w.jobs.created == [] and w.jobs.reset_calls == []
