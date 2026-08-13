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
    "main",
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


def ensure_index(
    w: Any = None,
    settings: SearchSettings | None = None,
    *,
    wait: bool = True,
    timeout_seconds: float = INDEX_TIMEOUT_SECONDS,
) -> Any:
    """Return the Delta Sync index, creating it only if it does not exist.

    Never re-creates: dropping a populated index throws away every embedding, and the rebuild is
    slow and billable. If the index exists but is misconfigured, delete it deliberately by hand.
    """
    w = _client(w)
    settings = settings or search_settings()

    try:
        existing = w.vector_search_indexes.get_index(index_name=settings.index_fqn)
    except NotFound:
        existing = None

    if existing is not None:
        log.info("search index already exists name=%s", settings.index_fqn)
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
) -> dict:
    """Create whatever is missing, then trigger the first sync. Safe to re-run.

    Returns a summary dict for notebook display.
    """
    w = _client(w)
    settings = search_settings(config)

    ensure_endpoint(w, settings, wait=wait)
    ensure_index(w, settings, wait=wait)

    if sync:
        trigger_sync(w, settings)
        if wait:
            wait_until_ready(w, settings.index_fqn)

    status = w.vector_search_indexes.get_index(index_name=settings.index_fqn).status
    return {
        "endpoint": settings.endpoint_name,
        "index": settings.index_fqn,
        "source_table": settings.source_table_fqn,
        "primary_key": settings.primary_key,
        "ready": bool(getattr(status, "ready", False)),
        "indexed_row_count": getattr(status, "indexed_row_count", None),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main())
