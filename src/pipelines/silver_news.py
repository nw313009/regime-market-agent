"""``bronze.news_raw`` -> ``silver.news_articles`` (spec A-3).

Payload shapes here are VERIFIED against a live Massive response, not inferred.

Schema: ``article_id``, ``ticker``, ``published_at``, ``title``, ``description``,
``publisher``, ``sentiment_label``, ``sentiment_score``, ``sentiment_reasoning``,
``embedding_text``, ``article_url``.

EXPLODE FROM ``insights``, NOT ``tickers``. Each article carries both a ``tickers`` array and
an ``insights`` array of ``{ticker, sentiment, sentiment_reasoning}``. Sentiment exists only
inside ``insights``, so the explode produces one row per (article, insight), taking BOTH the
ticker and its sentiment from the insight.

That explode already happened at ingestion (A-2), so bronze rows are one per (article, insight)
and this build is a projection, not a flattening. The rule still governs the result: a ticker
listed in ``tickers`` with no corresponding insight produces NO row, and ``bronze.news_raw``
keeps the raw array in ``article_tickers`` so the omission stays auditable. Observed live,
``insights`` tickers were a subset of ``tickers`` in 10/10 articles with zero mismatches, so a
strict subset is normal input, not an anomaly to repair.

Field mapping::

    article_id          <- bronze.article_id          (stable 64-char hex digest)
    published_at        <- bronze.published_utc       (ISO-8601 UTC, e.g. "2026-08-10T02:15:00Z")
    publisher           <- bronze.publisher_name      (publisher.name, flattened at ingestion)
    ticker              <- bronze.ticker              (insights[].ticker)
    sentiment_label     <- bronze.sentiment           (raw: positive/neutral/negative)
    sentiment_reasoning <- bronze.sentiment_reasoning (raw text, for agent/UI display)
    title, description, article_url pass through unchanged.

``sentiment_score`` is DERIVED here, not ingested. Massive returns no numeric score - insight
keys are exactly ``{ticker, sentiment, sentiment_reasoning}``. Map positive to +1, neutral to
0, negative to -1; any unrecognized label maps to 0 AND logs a warning, so a new vendor label
degrades to neutral rather than failing the build or vanishing silently. ``daily_features.s_t``
consumes ``sentiment_score`` unchanged (A-4).

The mapping lives in :data:`SENTIMENT_SCORES` and the SQL CASE expression is GENERATED from that
dict by :func:`sentiment_score_sql`, so the Python rule and the executed SQL cannot disagree.
The warning cannot be raised from inside a SQL CASE, so :func:`main` scans the distinct labels
in bronze first and warns once per unrecognized label, with a row count.

``embedding_text`` = ``title + "\\n" + description``. It is the embedding source column for the
AI Search Delta Sync index. A missing or blank description yields the title alone, with no
trailing separator, and an article with neither yields NULL rather than an empty string.

MERGE on the composite key ``(article_id, ticker)``.

Change Data Feed must be enabled at table creation, because the AI Search index reads the Delta
CDF::

    TBLPROPERTIES (delta.enableChangeDataFeed = true)

An index that is empty after a sync usually means this property was missing when the table was
created. ``tests/test_silver.py`` asserts the DDL carries it.

Timestamps: ``published_utc`` is an ISO-8601 string, NOT epoch-milliseconds - that applies to
the aggregates ``t`` field instead. It is parsed here and stored as an instant; mapping it to a
trading session (with the next-session rule for articles published while the market is closed)
belongs to A-4, not to this table.

STRUCTURE. The transform runs in Spark (spec rule 4), while each derivation also exists as a
pure Python function - :func:`sentiment_score`, :func:`unknown_sentiment_labels`,
:func:`build_embedding_text` - and those are what the unit tests pin.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from src.pipelines import (
    STATUS_FAILED,
    RunRecord,
    latest_per_key_sql,
    merge_select,
    new_run_id,
    qualified,
    record_run,
    require_table,
    truncate_error,
    utc_now,
)

log = logging.getLogger(__name__)

TASK_NAME = "build_silver_news"
SOURCE_TABLE = "bronze.news_raw"
TARGET_TABLE = "silver.news_articles"
MERGE_KEYS = ("article_id", "ticker")

#: The only recognized vendor labels, and the score each maps to. Single source of truth: the
#: SQL CASE expression is generated from this dict.
SENTIMENT_SCORES = {"positive": 1, "neutral": 0, "negative": -1}

#: Score for a label outside SENTIMENT_SCORES. Degrade to neutral, never drop the row.
UNKNOWN_SENTIMENT_SCORE = 0

EMBEDDING_TEXT_SEPARATOR = "\n"

#: Normalized label expression, shared by the score CASE and the unknown-label scan so both
#: judge the same string.
NORMALIZED_LABEL_EXPR = "lower(trim(sentiment))"

#: ``concat_ws`` skips NULL inputs, and ``nullif(trim(x), '')`` turns a blank into NULL, so this
#: is exactly the :func:`build_embedding_text` rule: join the non-empty parts with a newline.
#: The outer ``nullif`` returns NULL when both parts are missing, because ``concat_ws`` would
#: otherwise produce an empty string and the AI Search index would embed nothing.
EMBEDDING_TEXT_EXPR = (
    "nullif(concat_ws('\\n', nullif(trim(title), ''), nullif(trim(description), '')), '')"
)

#: Parse the raw ISO-8601 string, per spec. ``source_timestamp`` is the same instant, already
#: parsed in Python at ingestion, so it is an exact fallback if Spark ever rejects a vendor form.
PUBLISHED_AT_EXPR = "coalesce(try_to_timestamp(published_utc), source_timestamp)"

#: Latest ingestion wins if the same (article_id, ticker) was written twice.
DEDUPE_ORDER_BY = "ingested_at DESC"
DEDUPE_PARTITION_BY = ("article_id", "ticker")


def sentiment_score_sql(column: str = "sentiment") -> str:
    """Generate the CASE expression that maps a raw label to its score.

    Generated from :data:`SENTIMENT_SCORES` rather than written out, so adding a label to the
    dict changes the executed SQL and the Python function together.
    """
    branches = " ".join(
        f"WHEN '{label}' THEN {score}" for label, score in SENTIMENT_SCORES.items()
    )
    return f"CASE lower(trim({column})) {branches} ELSE {UNKNOWN_SENTIMENT_SCORE} END"


#: Ordered ``(column, expression)`` pairs. Column order matches the table DDL.
PROJECTIONS = (
    ("article_id", "article_id"),
    ("ticker", "ticker"),
    ("published_at", PUBLISHED_AT_EXPR),
    ("title", "title"),
    ("description", "description"),
    ("publisher", "publisher_name"),
    ("sentiment_label", "sentiment"),
    ("sentiment_score", sentiment_score_sql()),
    ("sentiment_reasoning", "sentiment_reasoning"),
    ("embedding_text", EMBEDDING_TEXT_EXPR),
    ("article_url", "article_url"),
)

NEWS_ARTICLE_COLUMNS = tuple(name for name, _ in PROJECTIONS)


# ------------------------------------------------------------------ pure functions


def normalize_sentiment_label(label: Any) -> str | None:
    """Lower-case and trim a raw label. Blank or missing becomes ``None``."""
    if label is None:
        return None
    text = str(label).strip().lower()
    return text or None


def sentiment_score(label: Any) -> int:
    """Map a raw vendor label to +1 / 0 / -1, with 0 for anything unrecognized.

    Logs a WARNING for an unrecognized label, which is the A-3 requirement: a new vendor label
    must degrade to neutral loudly, never fail the build and never be silently dropped. A
    missing label is logged at DEBUG instead — there is no new label to learn about.
    """
    normalized = normalize_sentiment_label(label)
    if normalized is None:
        log.debug("news row has no sentiment label, scoring 0")
        return UNKNOWN_SENTIMENT_SCORE
    if normalized not in SENTIMENT_SCORES:
        log.warning(
            "unrecognized sentiment label, degrading to neutral label=%r score=%d",
            label,
            UNKNOWN_SENTIMENT_SCORE,
        )
        return UNKNOWN_SENTIMENT_SCORE
    return SENTIMENT_SCORES[normalized]


def unknown_sentiment_labels(labels: Iterable[Any]) -> list[str]:
    """Distinct normalized labels that are not in :data:`SENTIMENT_SCORES`, sorted.

    Missing and blank labels are excluded: they are not new vendor vocabulary, and warning on
    every null row would bury the case that matters.
    """
    unknown = {
        normalized
        for normalized in (normalize_sentiment_label(label) for label in labels)
        if normalized is not None and normalized not in SENTIMENT_SCORES
    }
    return sorted(unknown)


def build_embedding_text(title: Any, description: Any) -> str | None:
    """``title`` + newline + ``description``, skipping whichever part is missing.

    Mirrors :data:`EMBEDDING_TEXT_EXPR` exactly, including the NULL-not-empty-string result when
    both parts are absent. A blank description must not leave a trailing newline: the separator
    would be embedded as content by the AI Search index.
    """
    parts = [
        str(part).strip()
        for part in (title, description)
        if part is not None and str(part).strip()
    ]
    return EMBEDDING_TEXT_SEPARATOR.join(parts) if parts else None


def build_source_sql(catalog: str) -> str:
    """The deduplicated SELECT that feeds the MERGE."""
    return latest_per_key_sql(
        qualified(catalog, SOURCE_TABLE),
        PROJECTIONS,
        DEDUPE_PARTITION_BY,
        DEDUPE_ORDER_BY,
    )


# --------------------------------------------------------------------- entry point


def _warn_about_unknown_labels(spark: Any, source_fqn: str) -> list[str]:
    """Warn once per unrecognized label in bronze, with its row count.

    A SQL CASE cannot log, so the warning the spec requires is raised here, from one grouped
    scan of the source table, before the MERGE runs. Returns the labels for the run summary.
    """
    rows = spark.sql(
        f"SELECT {NORMALIZED_LABEL_EXPR} AS label, count(*) AS n FROM {source_fqn} GROUP BY 1"
    ).collect()
    counts = {row["label"]: int(row["n"]) for row in rows}

    unknown = unknown_sentiment_labels(counts)
    for label in unknown:
        log.warning(
            "unrecognized sentiment label in bronze, scoring %d label=%r rows=%d",
            UNKNOWN_SENTIMENT_SCORE,
            label,
            counts[label],
        )
    return unknown


def main(spark: Any, config: Mapping) -> dict:
    """Build ``silver.news_articles`` from ``bronze.news_raw``.

    Callable identically from a workflow task and from a notebook cell::

        from src.pipelines import silver_news
        silver_news.main(spark, config)

    Returns a summary dict for notebook display, including any unrecognized sentiment labels
    seen. The whole of bronze is rebuilt on every run: the tables are tiny, and a MERGE on
    ``(article_id, ticker)`` makes the rebuild idempotent.

    Exactly one ``bronze.ingestion_runs`` row is written per call, on success and on failure,
    the same as the ingestion tasks.
    """
    catalog = str(config["catalog"])
    source_fqn = qualified(catalog, SOURCE_TABLE)
    target_fqn = qualified(catalog, TARGET_TABLE)
    run = RunRecord(run_id=new_run_id(), task=TASK_NAME, started_at=utc_now())
    unknown_labels: list[str] = []

    try:
        require_table(spark, source_fqn)
        require_table(spark, target_fqn)

        unknown_labels = _warn_about_unknown_labels(spark, source_fqn)
        run.rows_written = merge_select(spark, target_fqn, build_source_sql(catalog), MERGE_KEYS)
    except BaseException as exc:
        run.status = STATUS_FAILED
        run.error = truncate_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finished_at = utc_now()
        record_run(spark, catalog, run)

    log.info(
        "%s complete run_id=%s rows_merged=%d target=%s",
        TASK_NAME,
        run.run_id,
        run.rows_written,
        target_fqn,
    )
    return {
        "task": TASK_NAME,
        "run_id": run.run_id,
        "source": source_fqn,
        "target": target_fqn,
        "rows_merged": run.rows_written,
        "unknown_sentiment_labels": unknown_labels,
    }


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
