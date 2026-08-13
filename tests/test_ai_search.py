"""AI Search setup tests (spec C-1). No workspace: the SDK clients are faked.

What is worth testing in a setup script is not "does it call the API" but the handful of choices
that are expensive or invisible to get wrong:

- IT ASKS BEFORE IT CREATES. Re-running the script must not re-create the index; that would
  discard every embedding and re-embed the whole table.
- THE INDEX SPEC IS THE ONE THE SPEC ASKED FOR: Delta Sync, TRIGGERED, managed embeddings from
  ``embedding_text``, keyed on ``doc_id``.
- THE SYNCED COLUMNS EXIST IN THE SOURCE TABLE THE SHIPPED CONFIG NAMES. A typo'd column here —
  or a source table that does not carry ``doc_id`` and ``embedding_text`` — fails at sync time in
  the workspace, hours later and far from the edit. Checked against the DDL, and against the real
  config.yaml rather than the fixture below, because the fixture cannot drift into production.
- WAITING TERMINATES. A wait helper with no deadline is how a notebook hangs overnight.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    IndexSubtype,
    PipelineType,
    VectorIndexType,
)

from setup.create_ai_search import (
    SYNCED_COLUMNS,
    ensure_endpoint,
    ensure_index,
    main,
    search_settings,
    trigger_sync,
    wait_until_ready,
)
from src.pipelines.news_recent import COLUMNS as NEWS_RECENT_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL = (REPO_ROOT / "setup" / "create_delta_tables.sql").read_text(encoding="utf-8")
SHIPPED_CONFIG = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

CONFIG = {
    "catalog": "market_intel",
    "search": {
        "endpoint_name": "market-intel-search",
        "index": "silver.news_index",
        "source_table": "silver.news_recent",
        "primary_key": "doc_id",
        "embedding_source_column": "embedding_text",
        "embedding_model_endpoint": "databricks-gte-large-en",
        "top_k": 5,
    },
}

SETTINGS = search_settings(CONFIG)


def ddl_columns(table: str) -> list[str]:
    """Column names declared for ``catalog.schema.table`` in create_delta_tables.sql."""
    block = re.search(
        rf"CREATE TABLE IF NOT EXISTS \S*{re.escape(table)}\s*\((.*?)\n\)",
        DDL,
        re.DOTALL,
    )
    assert block is not None, f"no DDL found for {table}"
    return re.findall(r"^\s{2}(\w+)\s", block.group(1), re.MULTILINE)


# --------------------------------------------------------------------------- fakes


class FakeStatus:
    def __init__(self, ready=True, indexed_row_count=28000, message="Online"):
        self.ready = ready
        self.indexed_row_count = indexed_row_count
        self.message = message


class FakeIndex:
    def __init__(self, name, status=None):
        self.name = name
        self.status = status or FakeStatus()


class FakeWaiter:
    def __init__(self, value):
        self.value = value
        self.waited = None

    def result(self, timeout=None):
        self.waited = timeout
        return self.value


class FakeEndpoints:
    def __init__(self, existing=None):
        self.existing = existing
        self.created: list[dict] = []

    def get_endpoint(self, endpoint_name):
        if self.existing is None:
            raise NotFound(f"endpoint {endpoint_name} does not exist")
        return self.existing

    def create_endpoint(self, **kwargs):
        self.created.append(kwargs)
        self.existing = {"name": kwargs["name"]}
        return FakeWaiter(self.existing)


class FakeIndexes:
    def __init__(self, existing=None, statuses=None):
        self.existing = existing
        self.created: list[dict] = []
        self.synced: list[str] = []
        self.statuses = list(statuses or [])
        self.get_calls = 0

    def get_index(self, index_name):
        self.get_calls += 1
        if self.existing is None:
            raise NotFound(f"index {index_name} does not exist")
        if self.statuses:
            return FakeIndex(index_name, self.statuses.pop(0))
        return self.existing

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        self.existing = FakeIndex(kwargs["name"])
        return self.existing

    def sync_index(self, index_name):
        self.synced.append(index_name)


class FakeWorkspace:
    def __init__(self, endpoints=None, indexes=None):
        self.vector_search_endpoints = endpoints or FakeEndpoints()
        self.vector_search_indexes = indexes or FakeIndexes()


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


# ======================================================================== settings


def test_settings_qualify_the_names_with_the_configured_catalog():
    assert SETTINGS.index_fqn == "market_intel.silver.news_index"
    assert SETTINGS.source_table_fqn == "market_intel.silver.news_recent"
    assert SETTINGS.endpoint_name == "market-intel-search"


def test_a_second_catalog_needs_no_other_edit():
    other = search_settings({**CONFIG, "catalog": "market_intel_dev"})

    assert other.index_fqn == "market_intel_dev.silver.news_index"
    assert other.source_table_fqn == "market_intel_dev.silver.news_recent"


def test_the_key_is_doc_id_because_the_table_is_grained_on_two_columns():
    # article_id is not unique here: an article with three insights is three rows (A-3).
    assert SETTINGS.primary_key == "doc_id"
    assert "doc_id" in NEWS_RECENT_COLUMNS


def test_every_synced_column_exists_in_the_window_the_pipeline_writes():
    missing = sorted(set(SYNCED_COLUMNS) - set(NEWS_RECENT_COLUMNS))

    assert missing == [], f"columns_to_sync names columns news_recent does not carry: {missing}"


# --------------------------------------------------- the SHIPPED config, not the fixture above


def test_the_configured_source_table_carries_every_synced_column():
    """The fixture above cannot drift into production; config.yaml can.

    A source table missing ``doc_id`` or ``embedding_text`` is accepted by create_index and fails
    at sync time in the workspace, which is a slow and confusing way to learn it.
    """
    source = SHIPPED_CONFIG["search"]["source_table"]
    columns = ddl_columns(source)
    missing = sorted(set(SYNCED_COLUMNS) - set(columns))

    assert missing == [], f"{source} is missing {missing}"
    assert SHIPPED_CONFIG["search"]["primary_key"] in columns
    assert SHIPPED_CONFIG["search"]["embedding_source_column"] in columns


def test_the_configured_source_table_has_change_data_feed_on():
    """Delta Sync reads the CDF. Without it the index syncs to zero rows and says nothing."""
    source = SHIPPED_CONFIG["search"]["source_table"]
    # To the next statement, not to the next ";" — a table COMMENT is allowed to contain one.
    block = DDL.split(f"CREATE TABLE IF NOT EXISTS market_intel.{source}", 1)[1]
    block = block.split("CREATE TABLE IF NOT EXISTS", 1)[0]

    assert "delta.enableChangeDataFeed = true" in block


def test_the_index_is_built_on_the_window_so_it_inherits_the_retention():
    """An index over the archive grows forever; over the window it is trimmed by the refresh."""
    assert SHIPPED_CONFIG["search"]["source_table"] == "silver.news_recent"


def test_the_columns_the_search_tool_returns_are_synced():
    from src.agent.tools import NEWS_COLUMNS

    assert set(NEWS_COLUMNS) <= set(SYNCED_COLUMNS)


# ======================================================================== endpoint


def test_an_existing_endpoint_is_returned_untouched():
    endpoints = FakeEndpoints(existing={"name": "market-intel-search"})
    w = FakeWorkspace(endpoints=endpoints)

    result = ensure_endpoint(w, SETTINGS)

    assert result == {"name": "market-intel-search"}
    assert endpoints.created == []


def test_a_missing_endpoint_is_created_as_standard():
    endpoints = FakeEndpoints(existing=None)
    w = FakeWorkspace(endpoints=endpoints)

    ensure_endpoint(w, SETTINGS, wait=False)

    assert endpoints.created == [
        {"name": "market-intel-search", "endpoint_type": EndpointType.STANDARD}
    ]


def test_waiting_for_the_endpoint_resolves_the_waiter():
    w = FakeWorkspace(endpoints=FakeEndpoints(existing=None))

    result = ensure_endpoint(w, SETTINGS, wait=True, timeout_seconds=60)

    assert result == {"name": "market-intel-search"}


# =========================================================================== index


def test_an_existing_index_is_never_recreated():
    # Re-creating drops the embeddings and re-embeds ~28k rows for nothing.
    indexes = FakeIndexes(existing=FakeIndex("market_intel.silver.news_index"))
    w = FakeWorkspace(indexes=indexes)

    ensure_index(w, SETTINGS)

    assert indexes.created == []


def test_a_missing_index_is_created_as_a_triggered_delta_sync_index():
    indexes = FakeIndexes(existing=None)
    w = FakeWorkspace(indexes=indexes)

    ensure_index(w, SETTINGS, wait=False)

    created = indexes.created[0]
    assert created["name"] == "market_intel.silver.news_index"
    assert created["endpoint_name"] == "market-intel-search"
    assert created["primary_key"] == "doc_id"
    assert created["index_type"] is VectorIndexType.DELTA_SYNC
    assert created["index_subtype"] is IndexSubtype.HYBRID

    spec = created["delta_sync_index_spec"]
    assert spec.source_table == "market_intel.silver.news_recent"
    assert spec.pipeline_type is PipelineType.TRIGGERED
    assert list(spec.columns_to_sync) == list(SYNCED_COLUMNS)


def test_the_embeddings_are_managed_from_the_source_column():
    # No embedding is generated or stored by this repository; Databricks computes them at sync.
    indexes = FakeIndexes(existing=None)

    ensure_index(FakeWorkspace(indexes=indexes), SETTINGS, wait=False)

    columns = indexes.created[0]["delta_sync_index_spec"].embedding_source_columns
    assert len(columns) == 1
    assert columns[0].name == "embedding_text"
    assert columns[0].embedding_model_endpoint_name == "databricks-gte-large-en"


# ============================================================= sync and readiness


def test_trigger_sync_targets_the_configured_index():
    indexes = FakeIndexes(existing=FakeIndex("market_intel.silver.news_index"))
    w = FakeWorkspace(indexes=indexes)

    assert trigger_sync(w, SETTINGS) == "market_intel.silver.news_index"
    assert indexes.synced == ["market_intel.silver.news_index"]


def test_wait_until_ready_polls_until_the_index_reports_ready():
    indexes = FakeIndexes(
        existing=FakeIndex("i"),
        statuses=[FakeStatus(ready=False), FakeStatus(ready=False), FakeStatus(ready=True)],
    )
    clock = FakeClock()

    status = wait_until_ready(
        FakeWorkspace(indexes=indexes),
        "market_intel.silver.news_index",
        poll_seconds=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert status.ready is True
    assert clock.slept == [5, 5]


def test_waiting_gives_up_rather_than_hanging_a_notebook():
    indexes = FakeIndexes(existing=FakeIndex("i"), statuses=[FakeStatus(ready=False)] * 50)
    clock = FakeClock()

    with pytest.raises(TimeoutError, match="was not ready"):
        wait_until_ready(
            FakeWorkspace(indexes=indexes),
            "market_intel.silver.news_index",
            timeout_seconds=30,
            poll_seconds=10,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


# ============================================================================ main


def test_main_creates_what_is_missing_then_syncs():
    endpoints = FakeEndpoints(existing=None)
    indexes = FakeIndexes(existing=None)
    w = FakeWorkspace(endpoints=endpoints, indexes=indexes)

    summary = main(w, CONFIG, wait=False)

    assert endpoints.created and indexes.created
    assert indexes.synced == ["market_intel.silver.news_index"]
    assert summary["index"] == "market_intel.silver.news_index"
    assert summary["primary_key"] == "doc_id"
    assert summary["ready"] is True


def test_a_second_run_changes_nothing_but_the_sync():
    endpoints = FakeEndpoints(existing={"name": "market-intel-search"})
    indexes = FakeIndexes(existing=FakeIndex("market_intel.silver.news_index"))
    w = FakeWorkspace(endpoints=endpoints, indexes=indexes)

    main(w, CONFIG, wait=False)

    assert endpoints.created == []
    assert indexes.created == []
    assert indexes.synced == ["market_intel.silver.news_index"]
