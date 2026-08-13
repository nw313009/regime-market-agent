"""AI Search setup (spec C-1).

Creates:

- One vector search endpoint, STANDARD tier.
- A Delta Sync index ``market_intel.silver.news_index`` over ``silver.news_recent``, with
  ``embedding_text`` as the embedding source column (managed embeddings), sync mode
  TRIGGERED.

The ``sync_news_index`` workflow task triggers the sync, after ``refresh_news_recent`` has moved
the window it indexes.

Query path used by ``search_market_news``: hybrid search, filtered by ticker, top_k around 5.

THE SOURCE IS THE WINDOW, NOT THE ARCHIVE. ``search.source_table`` was ``silver.news_articles``
until C-6 gave the rolling window a table of its own. Indexing the archive means embedding every
article ever ingested and holding those vectors forever; indexing the window means the index
inherits its retention, since a row that ages out is deleted there and the next sync drops it
here. It also makes the agent's corpus and the Market Research news list the same set of articles.
The cost is that news older than ``news_recent.window_days`` is not retrievable, which is a config
change plus a backfill rather than a code change.

The source table must have had ``delta.enableChangeDataFeed = true`` set at creation time, since
the Delta Sync index reads the CDF. An index that is empty after a sync is this property missing,
not a sync bug. ``silver.news_recent`` is a cache, so the repair is DROP TABLE and re-run
``refresh_news_recent``.

Financial news text is the project's required unstructured-data path. Retrieval supplies
evidence; it never generates the numerical forecast.

RUN IT FROM A NOTEBOOK::

    %pip install -r requirements-databricks.txt
    from setup import create_ai_search
    create_ai_search.main()            # creates what is missing, waits, triggers the first sync

IDEMPOTENT, and in the only way that matters: it asks before it creates. Both ``ensure_``
functions look the resource up first and return the existing one untouched, so a second run
neither fails nor rebuilds the index — re-creating it would drop the embeddings and re-embed the
whole window for nothing. ``ResourceAlreadyExists`` is caught as well, for the case where two runs
overlap.

BUT "IT EXISTS" IS NOT "IT IS RIGHT", which this script learned the expensive way. When
``search.source_table`` moved from ``news_articles`` to ``news_recent``, re-running ``main()``
found the old index BY NAME, returned it unexamined, synced it from the old table and reported
success; three separate things had to be wrong for that to be silent, and all three are fixed
here:

- ``ensure_index`` now compares the LIVE index against config (:func:`index_drift`) and refuses a
  mismatch, naming the differences and the delete call. It still never deletes anything itself.
- ``main`` builds its summary from ``get_index``, not from the settings it was handed. The old
  summary echoed config, so it reported the source table the operator WANTED regardless of what
  the index used — the one output that could have exposed the drift instead concealed it.
- :func:`narrate` gives this module's INFO lines a handler when nothing else has, because
  ``logging.basicConfig`` under ``__main__`` never runs on the notebook path and the whole run was
  silent.

PRIMARY KEY. An AI Search index takes exactly ONE key column, and the source is grained on
``(article_id, ticker)``. The key here is therefore the derived ``doc_id``
(``article_id:ticker``, added in the silver build at C-1 and carried into the window), not
``article_id``: keying on the article alone would let one row of a multi-ticker article win
arbitrarily, and a ticker-filtered search would then answer "no relevant news" for an article that
exists. The cost is that a 3-ticker article is embedded 3 times, which is the shape of the
insights explode anyway (A-3).

SDK SURFACE, verified against the installed databricks-sdk 0.125.0 rather than recalled:
``w.vector_search_endpoints`` has ``get_endpoint(endpoint_name)`` / ``create_endpoint(name,
endpoint_type)`` / ``wait_get_endpoint_vector_search_endpoint_online(...)``, and
``w.vector_search_indexes`` has ``get_index(index_name)`` / ``create_index(name, endpoint_name,
primary_key, index_type, delta_sync_index_spec=..., index_subtype=...)`` / ``sync_index`` /
``query_index``. The index status object exposes ``ready`` and ``indexed_row_count``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from databricks.sdk.errors import NotFound, ResourceAlreadyExists
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    IndexSubtype,
    PipelineType,
    VectorIndexType,
)

from src.llm import config_section, load_config

log = logging.getLogger(__name__)

__all__ = [
    "SearchSettings",
    "ensure_endpoint",
    "ensure_index",
    "index_drift",
    "main",
    "narrate",
    "search_settings",
    "trigger_sync",
    "wait_until_ready",
]

#: Columns carried into the index. The primary key and the embedding source column are always
#: synced whatever this says; the rest is exactly what ``search_market_news`` returns, and no
#: more. Every name here must exist in ``search.source_table``; tests/test_ai_search.py asserts
#: that against the source table the SHIPPED config names, parsed out of the DDL.
SYNCED_COLUMNS = (
    "doc_id",
    "article_id",
    "ticker",
    "title",
    "publisher",
    "published_at",
    "sentiment_label",
    "article_url",
    "embedding_text",
)

#: How long to wait for the endpoint to come ONLINE. Provisioning one takes minutes, not seconds.
ENDPOINT_TIMEOUT_SECONDS = 30 * 60

#: How long to wait for the index to report ready, and how often to ask. The first sync embeds
#: every row, so this is the slow one.
INDEX_TIMEOUT_SECONDS = 60 * 60
INDEX_POLL_SECONDS = 15.0


@dataclass(frozen=True)
class SearchSettings:
    """Everything about the index that comes from config, resolved to full names."""

    endpoint_name: str
    index_fqn: str
    source_table_fqn: str
    primary_key: str
    embedding_source_column: str
    embedding_model_endpoint: str
    top_k: int
    columns_to_sync: tuple[str, ...] = SYNCED_COLUMNS


def search_settings(config: Mapping[str, Any] | None = None) -> SearchSettings:
    """Resolve the ``search`` config section, qualifying table names with ``catalog``.

    Table names are stored catalog-relative for the same reason as everywhere else in the code:
    ``catalog`` is one key, and pointing the project at a second catalog must not mean editing
    every name.
    """
    source = config if config is not None else load_config()
    section = config_section("search", source)
    catalog = str(source["catalog"])

    return SearchSettings(
        endpoint_name=str(section["endpoint_name"]),
        index_fqn=f"{catalog}.{section['index']}",
        source_table_fqn=f"{catalog}.{section['source_table']}",
        primary_key=str(section["primary_key"]),
        embedding_source_column=str(section["embedding_source_column"]),
        embedding_model_endpoint=str(section["embedding_model_endpoint"]),
        top_k=int(section["top_k"]),
    )


def narrate(enabled: bool = True) -> None:
    """Make this module's INFO lines visible when nothing else will.

    THE FOOTGUN THIS CLOSES: every progress line here is log.info on a module logger, and
    logging.basicConfig only ran under ``if __name__ == "__main__"``. Imported into a notebook —
    the only way this is ever actually run — the root logger sits at WARNING and the entire run is
    silent, which is how a re-run that changed nothing looked exactly like a re-run that worked.

    Deliberately narrow: it does nothing if this logger already has a handler (so repeated calls
    cannot stack them), and nothing if INFO is already enabled AND the root has somewhere to put
    it. Checking only for root handlers would not be enough — a runtime that installs a handler at
    WARNING leaves us just as silent.
    """
    if not enabled or log.handlers:
        return
    if log.isEnabledFor(logging.INFO) and logging.getLogger().handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _client(w: Any = None) -> Any:
    """The caller's WorkspaceClient, or a default one.

    Imported lazily so that reading this module — which the tools and the tests do — does not
    require workspace credentials to be resolvable.
    """
    if w is not None:
        return w
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


# --------------------------------------------------------------------- endpoint


def ensure_endpoint(
    w: Any = None,
    settings: SearchSettings | None = None,
    *,
    wait: bool = True,
    timeout_seconds: float = ENDPOINT_TIMEOUT_SECONDS,
) -> Any:
    """Return the STANDARD search endpoint, creating it only if it does not exist."""
    w = _client(w)
    settings = settings or search_settings()

    try:
        existing = w.vector_search_endpoints.get_endpoint(endpoint_name=settings.endpoint_name)
    except NotFound:
        existing = None

    if existing is not None:
        log.info("search endpoint already exists name=%s", settings.endpoint_name)
        return existing

    log.info("creating search endpoint name=%s type=STANDARD", settings.endpoint_name)
    try:
        waiter = w.vector_search_endpoints.create_endpoint(
            name=settings.endpoint_name,
            endpoint_type=EndpointType.STANDARD,
        )
    except ResourceAlreadyExists:
        # Another run got there between the get and the create.
        return w.vector_search_endpoints.get_endpoint(endpoint_name=settings.endpoint_name)

    if not wait:
        return waiter

    # create_endpoint returns a Wait; resolving it polls until the endpoint is ONLINE.
    return waiter.result(timeout=timedelta(seconds=timeout_seconds))


# ------------------------------------------------------------------------ index


def _delta_sync_spec(settings: SearchSettings) -> DeltaSyncVectorIndexSpecRequest:
    """The Delta Sync specification: managed embeddings, TRIGGERED sync."""
    return DeltaSyncVectorIndexSpecRequest(
        source_table=settings.source_table_fqn,
        pipeline_type=PipelineType.TRIGGERED,
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name=settings.embedding_source_column,
                embedding_model_endpoint_name=settings.embedding_model_endpoint,
            )
        ],
        columns_to_sync=list(settings.columns_to_sync),
    )


def _normalize(value: Any) -> str:
    """A name or an enum, reduced to something comparable across the config/API boundary.

    Both sides are already fully qualified — ``search_settings`` prefixes the catalog and the API
    returns three-part names — so this is not doing the qualification. It absorbs the differences
    that are NOT drift: backticks, surrounding space, catalog casing, and an enum arriving as
    either ``IndexSubtype.HYBRID`` or the string ``"HYBRID"``. A guard that cries wolf over
    casing is a guard someone switches off.
    """
    if value is None:
        return ""
    value = getattr(value, "value", value)
    return str(value).replace("`", "").strip().casefold()


def index_drift(index: Any, settings: SearchSettings) -> list[str]:
    """Every way the LIVE index disagrees with config. Empty means it matches.

    THIS EXISTS BECAUSE "AN INDEX WITH THAT NAME EXISTS" IS NOT "THE INDEX IS RIGHT". When
    search.source_table moved from news_articles to news_recent, a re-run of main() found the old
    index by name, returned it untouched, synced it from the old table, and reported success —
    the config change had no effect and nothing said so.

    Strict on the three fields that decide whether retrieval is correct at all: the source table,
    the primary key, and the embedding source column. Missing information counts as drift, not as
    a pass — an object with no delta_sync_index_spec is either not a Delta Sync index or not
    something this function can vouch for, and "could not tell" must never read as "fine".

    Lenient on subtype, pipeline type and embedding endpoint, which are compared ONLY when the API
    returns them. They matter, but an older index that reports them as None is not misconfigured,
    and a false alarm on a working index costs more than the check is worth.
    """
    spec = getattr(index, "delta_sync_index_spec", None)
    if spec is None:
        return [
            "no delta_sync_index_spec on the index — it is not a Delta Sync index, or the API "
            "did not return its specification, and either way its source cannot be verified"
        ]

    drift: list[str] = []

    def compare(label: str, actual: Any, expected: Any, *, strict: bool) -> None:
        if not strict and _normalize(actual) == "":
            return
        if _normalize(actual) != _normalize(expected):
            drift.append(f"{label}: index has {actual!r}, config wants {expected!r}")

    compare("source_table", getattr(spec, "source_table", None), settings.source_table_fqn, strict=True)
    compare("primary_key", getattr(index, "primary_key", None), settings.primary_key, strict=True)

    columns = getattr(spec, "embedding_source_columns", None) or []
    if not columns:
        drift.append("embedding_source_columns: index has none, config wants "
                     f"{settings.embedding_source_column!r}")
    else:
        compare(
            "embedding_source_column",
            getattr(columns[0], "name", None),
            settings.embedding_source_column,
            strict=True,
        )
        compare(
            "embedding_model_endpoint",
            getattr(columns[0], "embedding_model_endpoint_name", None),
            settings.embedding_model_endpoint,
            strict=False,
        )

    compare("index_subtype", getattr(index, "index_subtype", None), IndexSubtype.HYBRID, strict=False)
    compare("pipeline_type", getattr(spec, "pipeline_type", None), PipelineType.TRIGGERED, strict=False)
    return drift


def ensure_index(
    w: Any = None,
    settings: SearchSettings | None = None,
    *,
    wait: bool = True,
    timeout_seconds: float = INDEX_TIMEOUT_SECONDS,
) -> Any:
    """Return the Delta Sync index, creating it only if it does not exist.

    Never re-creates, and never deletes: dropping a populated index throws away every embedding,
    and that is a decision for a person, not for a setup script that someone re-ran. What it does
    do is REFUSE an index that does not match config — see :func:`index_drift` — with an error
    naming the differences and the exact delete call, because a Delta Sync index's source table is
    immutable and there is no in-place repair to offer.
    """
    w = _client(w)
    settings = settings or search_settings()

    try:
        existing = w.vector_search_indexes.get_index(index_name=settings.index_fqn)
    except NotFound:
        existing = None

    if existing is not None:
        drift = index_drift(existing, settings)
        if drift:
            differences = "\n".join(f"  - {item}" for item in drift)
            raise ValueError(
                f"{settings.index_fqn} exists but does not match config:\n{differences}\n\n"
                "A Delta Sync index's source table cannot be changed in place. Verify the source "
                "table is populated, then delete the index deliberately and re-run this script:\n"
                f'  w.vector_search_indexes.delete_index(index_name="{settings.index_fqn}")'
            )
        log.info(
            "search index already exists and matches config name=%s source=%s",
            settings.index_fqn,
            settings.source_table_fqn,
        )
        return existing

    log.info(
        "creating Delta Sync index name=%s source=%s key=%s embedding=%s",
        settings.index_fqn,
        settings.source_table_fqn,
        settings.primary_key,
        settings.embedding_source_column,
    )
    try:
        index = w.vector_search_indexes.create_index(
            name=settings.index_fqn,
            endpoint_name=settings.endpoint_name,
            primary_key=settings.primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            # HYBRID, because search_market_news queries with query_type="HYBRID"; the SDK
            # documents VECTOR as unsupported.
            index_subtype=IndexSubtype.HYBRID,
            delta_sync_index_spec=_delta_sync_spec(settings),
        )
    except ResourceAlreadyExists:
        return w.vector_search_indexes.get_index(index_name=settings.index_fqn)

    if wait:
        wait_until_ready(w, settings.index_fqn, timeout_seconds=timeout_seconds)
    return index


def trigger_sync(w: Any = None, settings: SearchSettings | None = None) -> str:
    """Kick off a TRIGGERED sync. Returns the index name, for the workflow task's summary.

    This is the whole body of the ``sync_news_index`` task (C-6): the index pulls the new rows
    from the source table's Change Data Feed. It returns as soon as the sync is accepted — call
    :func:`wait_until_ready` if the caller actually needs the rows to be searchable.
    """
    w = _client(w)
    settings = settings or search_settings()

    w.vector_search_indexes.sync_index(index_name=settings.index_fqn)
    log.info("sync triggered index=%s", settings.index_fqn)
    return settings.index_fqn


def wait_until_ready(
    w: Any = None,
    index_name: str | None = None,
    *,
    timeout_seconds: float = INDEX_TIMEOUT_SECONDS,
    poll_seconds: float = INDEX_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    """Poll ``get_index`` until the index reports ready, and return its status.

    Hand-rolled rather than borrowed from the SDK because the index API has no waiter of its own
    (only the endpoint does). ``ready`` is what the query path needs: an index that exists but is
    not ready answers queries with an error, not with zero results.
    """
    w = _client(w)
    index_name = index_name or search_settings().index_fqn

    deadline = monotonic() + timeout_seconds
    status = None
    while True:
        status = w.vector_search_indexes.get_index(index_name=index_name).status
        if status is not None and status.ready:
            log.info(
                "index ready name=%s indexed_rows=%s",
                index_name,
                getattr(status, "indexed_row_count", None),
            )
            return status
        if monotonic() >= deadline:
            message = getattr(status, "message", None)
            raise TimeoutError(
                f"index {index_name} was not ready after {timeout_seconds:.0f}s: {message}"
            )
        log.info("waiting for index name=%s status=%s", index_name, getattr(status, "message", None))
        sleep(poll_seconds)


def main(
    w: Any = None,
    config: Mapping[str, Any] | None = None,
    *,
    wait: bool = True,
    sync: bool = True,
    verbose: bool = True,
) -> dict:
    """Create whatever is missing, then trigger the first sync. Safe to re-run.

    Returns a summary dict for notebook display. THE SUMMARY IS READ BACK FROM THE INDEX, not
    echoed from config: the previous version reported settings.source_table_fqn, so it printed the
    table config asked for whether or not the index used it, and confirmed a change that had not
    happened. Everything here except ``endpoint`` now comes from ``get_index``.
    """
    narrate(verbose)
    w = _client(w)
    settings = search_settings(config)

    ensure_endpoint(w, settings, wait=wait)
    ensure_index(w, settings, wait=wait)

    if sync:
        trigger_sync(w, settings)
        if wait:
            # NOTE: readiness is not sync completion. An index that was already ready reports
            # ready the instant a sync is triggered, so this returns immediately on a re-run and
            # indexed_row_count below may still be the previous number. It is the first build —
            # where the index is not ready until it has embedded the source — that this waits for.
            wait_until_ready(w, settings.index_fqn)

    index = w.vector_search_indexes.get_index(index_name=settings.index_fqn)
    spec = getattr(index, "delta_sync_index_spec", None)
    status = getattr(index, "status", None)
    drift = index_drift(index, settings)

    summary = {
        "endpoint": settings.endpoint_name,
        "index": getattr(index, "name", None) or settings.index_fqn,
        "source_table": getattr(spec, "source_table", None),
        "primary_key": getattr(index, "primary_key", None),
        "matches_config": not drift,
        "ready": bool(getattr(status, "ready", False)),
        "indexed_row_count": getattr(status, "indexed_row_count", None),
    }
    log.info("done %s", summary)
    return summary


if __name__ == "__main__":
    print(main())
