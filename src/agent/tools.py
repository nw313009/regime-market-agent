"""The four agent tools, each with a JSON-schema declaration (spec C-4).

Read tools (numerical and retrieval context)::

    get_market_forecast(ticker)
        -> the latest gold.forecast_runs row joined with the gold.regime_states row

    search_market_news(ticker, query, k=5)
        -> AI Search hybrid results, filtered by ticker

Write tools (Lakebase state)::

    update_watchlist(action: "add" | "remove", ticker)
        -> performs the Lakebase write, returns the new watchlist

    save_research_report(ticker, question, report_md)
        -> performs the Lakebase write, returns the new report id

The read tools return the numbers; they do not compute them. Retrieval does not generate the
forecast.

Write tools are the CDC demo path: the row lands in Lakebase and then arrives in the Delta
history table through Lakebase CDF.

NO SPARK. ``get_market_forecast`` reads Gold over the serverless SQL warehouse
(``src/database/delta.py``), because the Databricks App that hosts this agent has no
SparkSession. Values are bound as ``:name`` parameters, never formatted into the statement, and
every table is fully qualified from the configured catalog.

LAKEBASE IS IMPORTED LAZILY, inside the two write tools. ``psycopg`` aborts the serverless
notebook kernel at import (C-1), and this module is otherwise importable anywhere — a notebook
that wants the tool schemas, or a test that only exercises the read tools, must not die on an
import it never uses.

EVERY TOOL RETURNS A JSON-SERIALIZABLE DICT, including in the empty and the failed case. The
result goes back to the model as a tool message, so a dates-and-Decimals object would fail at
serialization time, and an exception escaping into the loop would end the turn instead of
letting the model say "there is no forecast for that ticker yet".
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from src.database import delta
from src.llm import config_section, load_config

log = logging.getLogger(__name__)

__all__ = [
    "TOOLS",
    "ToolError",
    "execute_tool",
    "get_market_forecast",
    "save_research_report",
    "search_market_news",
    "tool_specs",
    "update_watchlist",
]

FORECAST_TABLE = "gold.forecast_runs"
REGIME_TABLE = "gold.regime_states"

#: Same rule as ``src/database/lakebase.py``: a ticker arrives from model-generated text, so it
#: is validated before it reaches a query.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

#: Preference order when several models produced a forecast for the same latest date. The spec
#: says any ``model_used``, and this picks deterministically rather than arbitrarily: prefer the
#: news-aware model, then the plain Markov one, then GBM. Ordering by ``model_used`` alphabetically
#: would systematically surface ``gbm``, the bottom rung of the fallback ladder.
MODEL_PREFERENCE = ("news_markov", "markov", "gbm")

#: Columns pulled from the index, in the order the tool documents them. ``embedding_text`` backs
#: the snippet; ``doc_id`` is the index key and is returned for traceability.
NEWS_COLUMNS = (
    "doc_id",
    "title",
    "publisher",
    "published_at",
    "sentiment_label",
    "article_url",
    "embedding_text",
)

#: Snippet length. Long enough to show what the article is about, short enough that five of them
#: do not crowd the forecast numbers out of the model's attention.
SNIPPET_CHARS = 320

MAX_SEARCH_RESULTS = 20


class ToolError(RuntimeError):
    """A tool could not run. The agent turns this into an error result for the model."""


# ------------------------------------------------------------------ shared helpers


def _normalize_ticker(ticker: Any) -> str:
    symbol = str(ticker or "").strip().upper()
    if not _TICKER_RE.match(symbol):
        raise ToolError(
            f"{ticker!r} is not a ticker symbol (expected 1-10 characters, A-Z start, "
            "letters/digits/./- after)"
        )
    return symbol


def _catalog(config: Mapping[str, Any] | None = None) -> str:
    source = config if config is not None else load_config()
    return str(source["catalog"])


def _jsonable(value: Any) -> Any:
    """Convert a warehouse value into something ``json.dumps`` accepts.

    Dates and timestamps become ISO-8601 strings and Decimals become floats. The model reads
    these as text either way; the point is that serialization cannot fail halfway through a turn.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _row(row: Mapping[str, Any] | None, columns: Sequence[str]) -> dict:
    """Project a warehouse row onto ``columns``, JSON-safe, missing columns as ``None``."""
    source = row or {}
    return {column: _jsonable(source.get(column)) for column in columns}


# -------------------------------------------------------------- get_market_forecast

FORECAST_COLUMNS = (
    "forecast_id",
    "ticker",
    "as_of_date",
    "model_used",
    "horizon_days",
    "current_price",
    "price_p10",
    "price_p50",
    "price_p90",
    "return_p10",
    "return_p50",
    "return_p90",
    "prob_positive",
    "prob_loss_gt_5pct",
    "prob_low_vol",
    "prob_high_vol",
    "n_paths",
    "generated_at",
)

REGIME_COLUMNS = (
    "prob_low_vol",
    "prob_high_vol",
    "low_vol_mean",
    "low_vol_sigma",
    "high_vol_mean",
    "high_vol_sigma",
    "current_news_signal",
    "model_used",
)


def _model_rank_sql(column: str = "model_used") -> str:
    """CASE expression ranking the models, generated from :data:`MODEL_PREFERENCE`.

    Generated rather than written out so the constant and the executed SQL cannot drift. The
    model names are code constants, not input, so they are safe to compose into the statement —
    every value that comes from the caller is bound.
    """
    branches = " ".join(
        f"WHEN '{name}' THEN {rank}" for rank, name in enumerate(MODEL_PREFERENCE)
    )
    return f"CASE {column} {branches} ELSE {len(MODEL_PREFERENCE)} END"


def forecast_sql(catalog: str) -> str:
    """The latest forecast for one ticker: most recent ``as_of_date``, best available model."""
    return (
        f"SELECT * FROM {delta.qualified(catalog, FORECAST_TABLE)} "
        "WHERE ticker = :ticker "
        f"ORDER BY as_of_date DESC, {_model_rank_sql()} ASC, generated_at DESC "
        "LIMIT 1"
    )


def regime_sql(catalog: str) -> str:
    """The regime read matching a forecast, on ``(ticker, as_of_date)``."""
    return (
        f"SELECT * FROM {delta.qualified(catalog, REGIME_TABLE)} "
        "WHERE ticker = :ticker AND as_of_date = :as_of_date "
        "LIMIT 1"
    )


def get_market_forecast(
    ticker: str,
    *,
    config: Mapping[str, Any] | None = None,
    conn: Any = None,
) -> dict:
    """The latest stored forecast for a ticker, with the regime read it was built from.

    Returns ``{"found": False, ...}`` rather than raising when the ticker has no forecast yet:
    "no forecast has been computed for AMD" is a true answer the agent must be able to give, and
    it is the normal state for a ticker added to the watchlist minutes ago.
    """
    symbol = _normalize_ticker(ticker)
    catalog = _catalog(config)

    forecast = delta.query_one(forecast_sql(catalog), {"ticker": symbol}, conn=conn)
    if forecast is None:
        return {
            "found": False,
            "ticker": symbol,
            "message": (
                f"No forecast has been computed for {symbol}. The daily job writes one per "
                "ticker it tracks; a ticker added recently may not have been forecast yet."
            ),
        }

    as_of_date = forecast.get("as_of_date")
    regime = delta.query_one(
        regime_sql(catalog), {"ticker": symbol, "as_of_date": as_of_date}, conn=conn
    )

    result = {"found": True, **_row(forecast, FORECAST_COLUMNS)}
    result["regime"] = _row(regime, REGIME_COLUMNS) if regime is not None else None
    result["units"] = {
        "returns": "decimal fraction, so -0.031 is -3.1%",
        "probabilities": "share of simulated paths, in [0, 1]",
        "horizon": "trading days",
    }
    return result


# --------------------------------------------------------------- search_market_news


def _search_settings(config: Mapping[str, Any] | None = None) -> tuple[str, int]:
    """``(index_fqn, default_k)`` from the ``search`` config section.

    Resolved here rather than imported from ``setup/create_ai_search.py`` so that querying the
    index does not depend on the module that creates it — the app never runs setup code.
    ``tests/test_agent_tools.py`` asserts the two agree on the name.
    """
    source = config if config is not None else load_config()
    section = config_section("search", source)
    catalog = str(source["catalog"])
    return f"{catalog}.{section['index']}", int(section.get("top_k", 5))


def _index_client(client: Any = None) -> Any:
    """The vector search index API. Imported lazily; the app supplies its own in practice."""
    if client is not None:
        return client
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().vector_search_indexes


def _snippet(embedding_text: Any, title: Any) -> str:
    """The article body, trimmed.

    ``embedding_text`` is ``title + "\\n" + description`` (A-3), so the title prefix is dropped:
    repeating it inside the snippet wastes context and reads like padding.
    """
    text = str(embedding_text or "").strip()
    heading = str(title or "").strip()
    if heading and text.startswith(heading):
        text = text[len(heading) :].strip()
    if len(text) > SNIPPET_CHARS:
        text = text[:SNIPPET_CHARS].rstrip() + "..."
    return text


def _query_rows(response: Any) -> list[dict]:
    """Turn a ``QueryVectorIndexResponse`` into dicts keyed by column name.

    The response is columnar-ish: ``manifest.columns`` names the columns and
    ``result.data_array`` holds the rows as lists, so reading by position would silently
    mis-assign fields the day the index gains a column.
    """
    manifest = getattr(response, "manifest", None)
    result = getattr(response, "result", None)
    columns = [column.name for column in (getattr(manifest, "columns", None) or [])]
    rows = getattr(result, "data_array", None) or []
    return [dict(zip(columns, row, strict=False)) for row in rows]


def search_market_news(
    ticker: str,
    query: str,
    k: int = 5,
    *,
    config: Mapping[str, Any] | None = None,
    index_client: Any = None,
) -> dict:
    """Hybrid search over the indexed news for one ticker.

    Filtered by ticker rather than merely biased toward it: the agent attributes what it finds to
    a specific company, and a semantically similar article about a competitor would be attributed
    wrongly. Hybrid (vector plus keyword) because the questions mix themes with exact terms —
    "downside risk" needs the embedding, "H20 export licence" needs the keyword.
    """
    symbol = _normalize_ticker(ticker)
    text = str(query or "").strip()
    if not text:
        raise ToolError("query is empty — search_market_news needs something to search for")

    index_fqn, default_k = _search_settings(config)
    top_k = max(1, min(int(k or default_k), MAX_SEARCH_RESULTS))

    response = _index_client(index_client).query_index(
        index_name=index_fqn,
        columns=list(NEWS_COLUMNS),
        # The filter is a JSON document the service parses, not SQL, and the value is placed in
        # it by json.dumps rather than concatenated.
        filters_json=json.dumps({"ticker": symbol}),
        query_text=text,
        query_type="HYBRID",
        num_results=top_k,
    )

    articles = [
        {
            "title": row.get("title"),
            "publisher": row.get("publisher"),
            "published_at": _jsonable(row.get("published_at")),
            "sentiment_label": row.get("sentiment_label"),
            "article_url": row.get("article_url"),
            "snippet": _snippet(row.get("embedding_text"), row.get("title")),
        }
        for row in _query_rows(response)
    ]

    return {
        "ticker": symbol,
        "query": text,
        "count": len(articles),
        "articles": articles,
        "message": (
            f"No indexed news matched that query for {symbol}."
            if not articles
            else f"{len(articles)} article(s) retrieved for {symbol}."
        ),
    }


# ---------------------------------------------------------------- Lakebase writes


def _lakebase(module: Any = None) -> Any:
    """The Lakebase module, imported at call time.

    Lazy on purpose (C-1): importing ``psycopg`` on serverless compute aborts the kernel, and
    every other tool in this module must stay usable there.
    """
    if module is not None:
        return module
    from src.database import lakebase

    return lakebase


def _lakebase_kwargs(**values: Any) -> dict:
    """Drop the unset ones, so Lakebase's own defaults (the demo user and watchlist) apply."""
    return {key: value for key, value in values.items() if value is not None}


def update_watchlist(
    action: str,
    ticker: str,
    *,
    watchlist_id: str | None = None,
    added_by: str | None = None,
    lakebase: Any = None,
    conn: Any = None,
) -> dict:
    """Add or remove a ticker, and return the resulting watchlist.

    The CDC demo write. ``changed`` is reported honestly: adding a ticker that is already there
    is a no-op, and the agent has to confirm what actually happened rather than what was asked.
    """
    verb = str(action or "").strip().lower()
    if verb not in {"add", "remove"}:
        raise ToolError(f"action must be 'add' or 'remove', got {action!r}")

    symbol = _normalize_ticker(ticker)
    db = _lakebase(lakebase)
    common = _lakebase_kwargs(watchlist_id=watchlist_id, conn=conn)

    before = db.get_watchlist(**common)
    after = (
        db.add_ticker(symbol, **common, **_lakebase_kwargs(added_by=added_by))
        if verb == "add"
        else db.remove_ticker(symbol, **common)
    )

    return {
        "action": verb,
        "ticker": symbol,
        "watchlist": list(after),
        "changed": list(before) != list(after),
        "message": (
            f"{symbol} {'added to' if verb == 'add' else 'removed from'} the watchlist."
            if list(before) != list(after)
            else (
                f"{symbol} was already on the watchlist."
                if verb == "add"
                else f"{symbol} was not on the watchlist."
            )
        ),
    }


def save_research_report(
    ticker: str,
    question: str,
    report_md: str,
    *,
    forecast_id: str | None = None,
    user_id: str | None = None,
    lakebase: Any = None,
    conn: Any = None,
) -> dict:
    """Persist one written answer and return its id.

    ``forecast_id`` ties the report to the exact ``gold.forecast_runs`` row it quoted, so a saved
    report stays auditable against the numbers it was based on.
    """
    symbol = _normalize_ticker(ticker)
    body = str(report_md or "")
    if not body.strip():
        raise ToolError("report_md is empty — refusing to save an empty report")

    db = _lakebase(lakebase)
    report_id = db.save_report(
        symbol,
        str(question or ""),
        body,
        forecast_id=forecast_id,
        **_lakebase_kwargs(user_id=user_id, conn=conn),
    )

    return {
        "saved": True,
        "report_id": str(report_id),
        "ticker": symbol,
        "forecast_id": forecast_id,
        "message": f"Report saved for {symbol} (id {report_id}).",
    }


# ------------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Tool:
    """One callable plus the JSON schema the model sees for it."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., dict]

    def spec(self) -> dict:
        """The OpenAI-style function declaration ``call_model`` passes as ``tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def _schema(properties: Mapping[str, Any], required: Sequence[str]) -> dict:
    """A JSON Schema object. ``additionalProperties: false`` on every tool, so a hallucinated
    argument is rejected by the model's own validation instead of arriving here."""
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_TICKER_PROPERTY = {
    "type": "string",
    "description": "Stock ticker symbol, e.g. NVDA. Upper case; 1-10 characters.",
}

TOOLS: tuple[Tool, ...] = (
    Tool(
        name="get_market_forecast",
        description=(
            "Latest stored forecast for a ticker: horizon price and return percentiles, "
            "P(return > 0), P(loss worse than 5%), the regime probabilities behind them, and "
            "the model that produced them. Call this before making ANY quantitative claim. "
            "Returns found=false when no forecast exists for the ticker yet."
        ),
        parameters=_schema({"ticker": _TICKER_PROPERTY}, ["ticker"]),
        handler=get_market_forecast,
    ),
    Tool(
        name="search_market_news",
        description=(
            "Hybrid search over recent news articles about a ticker that this system has "
            "ingested and indexed. Returns title, publisher, published_at, sentiment_label, "
            "article_url and a snippet for each hit. Use it for every news claim, and cite the "
            "titles you use."
        ),
        parameters=_schema(
            {
                "ticker": _TICKER_PROPERTY,
                "query": {
                    "type": "string",
                    "description": "What to search for, in natural language.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many articles to return.",
                    "minimum": 1,
                    "maximum": MAX_SEARCH_RESULTS,
                    "default": 5,
                },
            },
            ["ticker", "query"],
        ),
        handler=search_market_news,
    ),
    Tool(
        name="update_watchlist",
        description=(
            "Add or remove a ticker from the user's watchlist. Performs the write immediately "
            "and returns the resulting watchlist. Confirm the result to the user."
        ),
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove"],
                    "description": "Whether to add or remove the ticker.",
                },
                "ticker": _TICKER_PROPERTY,
            },
            ["action", "ticker"],
        ),
        handler=update_watchlist,
    ),
    Tool(
        name="save_research_report",
        description=(
            "Save a written research answer for the user. Pass the finished markdown you would "
            "show them. Returns the new report id, which you must confirm."
        ),
        parameters=_schema(
            {
                "ticker": _TICKER_PROPERTY,
                "question": {
                    "type": "string",
                    "description": "The user's question, as asked.",
                },
                "report_md": {
                    "type": "string",
                    "description": "The full answer, in markdown.",
                },
            },
            ["ticker", "question", "report_md"],
        ),
        handler=save_research_report,
    ),
)

TOOLS_BY_NAME: Mapping[str, Tool] = {tool.name: tool for tool in TOOLS}


def tool_specs() -> list[dict]:
    """All four declarations, in the shape ``call_model(tools=...)`` expects."""
    return [tool.spec() for tool in TOOLS]


def execute_tool(name: str, arguments: Mapping[str, Any] | None = None, **context: Any) -> dict:
    """Run one tool by name. Raises :class:`ToolError` for an unknown name or bad arguments.

    ``context`` carries the injectables (``config``, ``conn``, ``index_client``, ``lakebase``)
    that the app and the tests supply; only the ones a given tool accepts are forwarded, so the
    agent can pass the whole context without knowing which tool needs what.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}")

    supplied = dict(arguments or {})
    allowed = set(tool.parameters.get("properties", {}))
    unexpected = sorted(set(supplied) - allowed)
    if unexpected:
        raise ToolError(f"{name} got unexpected argument(s): {', '.join(unexpected)}")

    missing = sorted(set(tool.parameters.get("required", ())) - set(supplied))
    if missing:
        raise ToolError(f"{name} is missing required argument(s): {', '.join(missing)}")

    return tool.handler(**supplied, **_accepted_context(tool.handler, context))


def _accepted_context(handler: Callable[..., dict], context: Mapping[str, Any]) -> dict:
    """The subset of ``context`` that ``handler`` names as a parameter.

    Matched by name rather than forwarded wholesale: the agent holds one context for all four
    tools, and handing ``index_client`` to a Lakebase write would be a TypeError at the worst
    possible moment.
    """
    parameters = inspect.signature(handler).parameters
    return {key: value for key, value in context.items() if key in parameters}
