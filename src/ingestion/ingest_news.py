"""News ingestion task: Massive news -> ``bronze.news_raw`` (spec A1.2, A-2).

For every ticker in the universe, fetch news published since the last stored article for that
ticker and MERGE into ``bronze.news_raw``.

Bronze rows keep the near-raw payload plus ``source``, ``ingested_at``, ``request_id``,
``ticker`` and ``source_timestamp``.

MERGE keys: ``(article_id, ticker)``.

EXPLODE FROM ``insights``, NOT from ``tickers`` (spec A-2/A-3). Each article carries both a
``tickers`` array and an ``insights`` array of ``{ticker, sentiment, sentiment_reasoning}``.
Sentiment exists ONLY inside insights, so the explode takes BOTH the ticker and the sentiment
from the same insight, and the bronze row already carries the per-insight ticker. Exploding
``tickers`` instead would manufacture rows with null sentiment and would leave the A-3 rule one
refactor from regressing. Deliberate consequence, accepted: a ticker listed in ``tickers`` with
no matching insight produces NO row. The raw array is preserved in ``article_tickers`` so that
decision stays auditable. Observed live: insight tickers are a strict subset of ``tickers``, so
treat strict-subset as normal rather than as an anomaly.

Insights for tickers OUTSIDE the requested universe are kept. An article fetched for NVDA that
also carries an SNDK insight yields both rows: bronze is near-raw, the ``(article_id, ticker)``
MERGE makes the overlap between per-ticker fetches idempotent, and discarding the row would
throw away data the vendor already sent.

``sentiment_score`` is NOT ingested. The payload has no numeric score; the ±1/0 mapping is
derived at silver build time (A-3).

INCREMENTAL WINDOW. Empty table for a ticker -> ``massive.backfill_start_date`` at 00:00Z.
Populated -> that ticker's own ``max(source_timestamp)``, inclusive, so an article published in
the same second as the watermark cannot fall through the gap; the MERGE absorbs the re-fetch.

STRUCTURE. The watermark arithmetic and the explode are pure functions, unit-tested without a
SparkSession (``tests/test_ingestion.py``). Spark appears only in :func:`main` and in the shared
write layer in ``src/ingestion/__init__.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from src.ingestion import (
    STATUS_FAILED,
    RunRecord,
    max_value,
    merge_rows,
    new_run_id,
    qualified,
    record_run,
    require_table,
    resolve_universe,
    truncate_error,
    utc_now,
)
from src.ingestion.massive_client import (
    REQUEST_ID_KEY,
    SOURCE,
    MassiveClient,
    env_secret_getter,
    redact_secrets,
)

log = logging.getLogger(__name__)

TASK_NAME = "ingest_news"
BRONZE_TABLE = "bronze.news_raw"
MERGE_KEYS = ("article_id", "ticker")

#: Column order shared by the row builder, the staging DataFrame and the table DDL.
NEWS_COLUMNS = (
    "article_id",
    "ticker",
    "source_timestamp",
    "published_utc",
    "title",
    "description",
    "author",
    "article_url",
    "image_url",
    "publisher_name",
    "publisher_homepage_url",
    "publisher_logo_url",
    "publisher_favicon_url",
    "sentiment",
    "sentiment_reasoning",
    "article_tickers",
    "keywords",
    "source",
    "ingested_at",
    "request_id",
)

NEWS_SCHEMA_DDL = (
    "article_id STRING, ticker STRING, source_timestamp TIMESTAMP, published_utc STRING, "
    "title STRING, description STRING, author STRING, article_url STRING, image_url STRING, "
    "publisher_name STRING, publisher_homepage_url STRING, publisher_logo_url STRING, "
    "publisher_favicon_url STRING, sentiment STRING, sentiment_reasoning STRING, "
    "article_tickers ARRAY<STRING>, keywords ARRAY<STRING>, source STRING, "
    "ingested_at TIMESTAMP, request_id STRING"
)


# ------------------------------------------------------------------ pure functions


def parse_published_utc(value: Any) -> datetime | None:
    """Parse the vendor's ISO-8601 UTC ``published_utc`` string into an aware datetime.

    News timestamps are ISO strings, unlike aggregates' epoch-milliseconds — the parsing step is
    per-source (spec A-3). Mapping the instant to a trading session happens in silver; bronze
    only needs a correct instant. Unparseable values return ``None`` so the caller can skip the
    row rather than merge a null key.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def news_fetch_start(watermark: datetime | None, backfill_start_date: date) -> str:
    """Resolve the ``published_after`` value for one ticker, as an ISO-8601 UTC string.

    No watermark (empty table for this ticker) -> the configured backfill start at midnight UTC.
    A watermark -> that instant, inclusive.
    """
    if watermark is None:
        return f"{backfill_start_date.isoformat()}T00:00:00Z"
    instant = watermark if watermark.tzinfo else watermark.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_news_rows(
    articles: Iterable[Mapping],
    ingested_at: datetime,
) -> list[dict]:
    """Explode articles into ``bronze.news_raw`` rows, one per (article, insight).

    Field mapping (verified live): ``id`` -> article_id, ``publisher.name`` -> publisher_name,
    ``published_utc`` -> source_timestamp (parsed) and published_utc (raw), ``insights[].ticker``
    -> ticker, ``insights[].sentiment`` -> sentiment (RAW label), ``insights[].sentiment_reasoning``
    -> sentiment_reasoning. Title, description, author, article_url and image_url pass through.

    Rows that cannot be keyed are dropped with a warning: no ``id``, no parseable
    ``published_utc``, or an insight with no ``ticker``. An article with no insights at all
    produces no rows, which is the A-3 rule rather than a failure.
    """
    rows: list[dict] = []
    for article in articles:
        article_id = article.get("id")
        if not article_id:
            log.warning("skipping news article without id publisher=%s", _publisher(article).get("name"))
            continue

        published_at = parse_published_utc(article.get("published_utc"))
        if published_at is None:
            log.warning(
                "skipping news article with unparseable published_utc article_id=%s", article_id
            )
            continue

        insights = article.get("insights") or []
        if not insights:
            # Not an error: without an insight there is no sentiment, and A-3 refuses to
            # manufacture a null-sentiment row from the tickers array.
            log.debug("news article has no insights, no rows emitted article_id=%s", article_id)
            continue

        publisher = _publisher(article)
        for insight in insights:
            ticker = (insight or {}).get("ticker")
            if not ticker:
                log.warning("skipping insight without ticker article_id=%s", article_id)
                continue
            rows.append(
                {
                    "article_id": str(article_id),
                    "ticker": str(ticker).strip().upper(),
                    "source_timestamp": published_at,
                    "published_utc": article.get("published_utc"),
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "author": article.get("author"),
                    "article_url": article.get("article_url"),
                    "image_url": article.get("image_url"),
                    "publisher_name": publisher.get("name"),
                    "publisher_homepage_url": publisher.get("homepage_url"),
                    "publisher_logo_url": publisher.get("logo_url"),
                    "publisher_favicon_url": publisher.get("favicon_url"),
                    # RAW label. The ±1/0 mapping is a silver concern (A-3).
                    "sentiment": insight.get("sentiment"),
                    "sentiment_reasoning": insight.get("sentiment_reasoning"),
                    "article_tickers": _as_str_list(article.get("tickers")),
                    "keywords": _as_str_list(article.get("keywords")),
                    "source": SOURCE,
                    "ingested_at": ingested_at,
                    "request_id": article.get(REQUEST_ID_KEY),
                }
            )
    return rows


def _publisher(article: Mapping) -> Mapping:
    """``publisher`` arrives as a nested dict; tolerate it being absent or a bare string."""
    publisher = article.get("publisher")
    if isinstance(publisher, Mapping):
        return publisher
    return {"name": publisher} if publisher else {}


def _as_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]


def _backfill_start_date(config: Mapping) -> date:
    raw = (config.get("massive") or {}).get("backfill_start_date")
    if not raw:
        raise ValueError(
            "config massive.backfill_start_date is required: it is the window used when a bronze "
            "table has no rows for a ticker yet."
        )
    return date.fromisoformat(str(raw))


# --------------------------------------------------------------------- entry point


def main(
    spark: Any,
    config: Mapping,
    *,
    client: Any | None = None,
    secret_getter: Any | None = None,
    watchlist: Sequence[str] | None = None,
) -> dict:
    """Run the news ingestion task.

    Callable identically from a workflow task and from a notebook cell::

        from src.ingestion import ingest_news
        ingest_news.main(spark, config,
                         secret_getter=lambda: dbutils.secrets.get("capstone",
                                                                   "massive_api_key"))

    Returns a summary dict (``run_id``, ``rows_written``, ``tickers``) for notebook display.
    Exactly one ``bronze.ingestion_runs`` row is written per call, on success and on failure.
    """
    catalog = str(config["catalog"])
    fqn = qualified(catalog, BRONZE_TABLE)
    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    backfill_start = _backfill_start_date(config)
    universe = resolve_universe(config, watchlist)

    try:
        require_table(spark, fqn)
        if client is None:
            client = MassiveClient(config["massive"], secret_getter or env_secret_getter())

        for ticker in universe:
            watermark = max_value(spark, fqn, "source_timestamp", "ticker", ticker)
            published_after = news_fetch_start(watermark, backfill_start)
            log.info(
                "fetching news ticker=%s published_after=%s mode=%s",
                ticker,
                published_after,
                "backfill" if watermark is None else "incremental",
            )
            articles = client.get_news(ticker, published_after)
            rows = build_news_rows(articles, ingested_at=utc_now())
            run.rows_written += merge_rows(
                spark,
                fqn,
                rows,
                columns=NEWS_COLUMNS,
                schema_ddl=NEWS_SCHEMA_DDL,
                keys=MERGE_KEYS,
            )
    except BaseException as exc:
        run.status = STATUS_FAILED
        # Redacted because the key is a query parameter: raw exception text can carry a
        # credential into a queryable table (A-1 security rule).
        run.error = truncate_error(redact_secrets(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s complete run_id=%s tickers=%d rows_written=%d",
        TASK_NAME,
        run.run_id,
        len(universe),
        run.rows_written,
    )
    return {"run_id": run.run_id, "rows_written": run.rows_written, "tickers": universe}


def _cli() -> None:
    """Wiring only: build a session, load config, call :func:`main`.

    Kept out of ``__main__`` so the module has no import-time side effects and the workflow task
    and the notebook both call the same function.
    """
    import argparse

    import yaml
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    logging.basicConfig(level=logging.INFO)
    main(SparkSession.builder.getOrCreate(), config)


if __name__ == "__main__":
    _cli()
