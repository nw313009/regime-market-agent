"""Agent tool tests (spec C-7).

- Write tools, as INTEGRATION tests against the real Lakebase: each of
  ``update_watchlist`` and ``save_research_report`` must produce the expected row. Mocking the
  database here would not test the thing that breaks.
- Read tools: ``get_market_forecast`` and ``search_market_news`` must return schema-valid
  payloads, including the empty case (a ticker with no forecast yet, a query with no matching
  news) — the agent has to be able to say "no relevant news" truthfully.
- Assert every tool's JSON-schema declaration matches its actual signature, since a drifted
  schema fails at model-call time rather than at import time.
- Assert all SQL is parameterized.

Manual end-to-end: run the A4 demo script once before declaring Checkpoint C frozen.

WHAT THE FAKES ARE FOR. The integration test above is real and opt-in (bottom of this file); it
needs a workspace, a warehouse, an index and a Postgres role, so it cannot be the only coverage.
The fakes below record what each tool ASKED FOR — the exact SQL text, the bound parameter dict,
the index filter — which is the part an integration test cannot assert precisely and the part
that breaks silently: a value interpolated into a query still returns the right row in a test and
is still wrong.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from setup.create_ai_search import search_settings
from src.agent import tools
from src.agent.tools import (
    MODEL_PREFERENCE,
    NEWS_COLUMNS,
    TOOLS,
    TOOLS_BY_NAME,
    ToolError,
    execute_tool,
    forecast_sql,
    get_market_forecast,
    regime_sql,
    save_research_report,
    search_market_news,
    tool_specs,
    update_watchlist,
)
from src.database import delta

CONFIG = {
    "catalog": "market_intel",
    "search": {
        "endpoint_name": "market-intel-search",
        "index": "silver.news_index",
        "source_table": "silver.news_articles",
        "primary_key": "doc_id",
        "embedding_source_column": "embedding_text",
        "embedding_model_endpoint": "databricks-gte-large-en",
        "top_k": 5,
    },
}

FORECAST_ROW = {
    "forecast_id": "f-1",
    "ticker": "NVDA",
    "generated_at": datetime(2026, 8, 10, 22, 41, tzinfo=timezone.utc),
    "as_of_date": date(2026, 8, 7),
    "horizon_days": 5,
    "model_used": "news_markov",
    "current_price": Decimal("182.50"),
    "price_p10": 171.2,
    "price_p50": 183.1,
    "price_p90": 195.4,
    "return_p10": -0.062,
    "return_p50": 0.003,
    "return_p90": 0.071,
    "prob_positive": 0.52,
    "prob_loss_gt_5pct": 0.11,
    "prob_low_vol": 0.31,
    "prob_high_vol": 0.69,
    "n_paths": 5000,
    "seed": 42,
    "model_version": "b6",
}

REGIME_ROW = {
    "ticker": "NVDA",
    "as_of_date": date(2026, 8, 7),
    "prob_low_vol": 0.31,
    "prob_high_vol": 0.69,
    "low_vol_mean": 0.0004,
    "low_vol_sigma": 0.011,
    "high_vol_mean": -0.0009,
    "high_vol_sigma": 0.034,
    "current_news_signal": -0.4,
    "model_used": "news_markov",
    "model_version": "b6",
}


# ------------------------------------------------------------------------- fakes


@dataclass
class Executed:
    sql: str
    params: object


class FakeCursor:
    """A databricks-sql-connector cursor: ``execute(sql, params)``, ``description``, ``fetchall``."""

    def __init__(self, connection):
        self._connection = connection
        self.description = ()
        self._rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._connection.executed.append(Executed(sql, params))
        rows = self._connection.next_rows()
        columns = sorted({key for row in rows for key in row})
        self.description = tuple((column,) + (None,) * 6 for column in columns)
        self._rows = [tuple(row.get(column) for column in columns) for row in rows]
        return self

    def fetchall(self):
        return list(self._rows)


class FakeWarehouse:
    """Hands back queued result sets and records every statement it was given."""

    def __init__(self, *result_sets):
        self.results = [list(rows) for rows in result_sets]
        self.executed: list[Executed] = []
        self.closed = False

    def next_rows(self):
        return self.results.pop(0) if self.results else []

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class FakeColumn:
    def __init__(self, name):
        self.name = name


class FakeQueryResponse:
    def __init__(self, columns, rows):
        self.manifest = type("Manifest", (), {"columns": [FakeColumn(c) for c in columns]})()
        self.result = type("Result", (), {"data_array": rows, "row_count": len(rows)})()


class FakeIndexClient:
    """Stands in for ``w.vector_search_indexes``."""

    def __init__(self, columns=NEWS_COLUMNS, rows=()):
        self.columns = list(columns)
        self.rows = [list(row) for row in rows]
        self.calls: list[dict] = []

    def query_index(self, **kwargs):
        self.calls.append(kwargs)
        return FakeQueryResponse(self.columns, self.rows)


class FakeLakebase:
    """Stands in for ``src.database.lakebase``, holding one watchlist in memory."""

    def __init__(self, watchlist=("NVDA", "MSFT")):
        self.watchlist = list(watchlist)
        self.reports: list[dict] = []
        self.calls: list[tuple] = []

    def get_watchlist(self, **kwargs):
        self.calls.append(("get_watchlist", kwargs))
        return sorted(self.watchlist)

    def add_ticker(self, ticker, **kwargs):
        self.calls.append(("add_ticker", ticker, kwargs))
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
        return sorted(self.watchlist)

    def remove_ticker(self, ticker, **kwargs):
        self.calls.append(("remove_ticker", ticker, kwargs))
        self.watchlist = [item for item in self.watchlist if item != ticker]
        return sorted(self.watchlist)

    def save_report(self, ticker, question, report_md, **kwargs):
        self.calls.append(("save_report", ticker, kwargs))
        self.reports.append(
            {"ticker": ticker, "question": question, "report_md": report_md, **kwargs}
        )
        return f"report-{len(self.reports)}"


def news_row(**overrides):
    """One index row, in NEWS_COLUMNS order."""
    values = {
        "doc_id": "abc:NVDA",
        "title": "Nvidia beats on data-center revenue",
        "publisher": "Reuters",
        "published_at": "2026-08-08T13:02:00Z",
        "sentiment_label": "positive",
        "article_url": "https://example.test/a",
        "embedding_text": "Nvidia beats on data-center revenue\nThe company reported...",
        **overrides,
    }
    return [values[column] for column in NEWS_COLUMNS]


# ============================================================== JSON-schema validity


def test_there_are_exactly_four_tools():
    assert sorted(TOOLS_BY_NAME) == [
        "get_market_forecast",
        "save_research_report",
        "search_market_news",
        "update_watchlist",
    ]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_every_declaration_is_a_valid_function_schema(tool):
    spec = tool.spec()

    assert spec["type"] == "function"
    assert spec["function"]["name"] == tool.name
    assert spec["function"]["description"].strip()

    schema = spec["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(schema["properties"])
    for name, prop in schema["properties"].items():
        assert prop.get("type"), f"{tool.name}.{name} has no type"
        assert prop.get("description", "").strip(), f"{tool.name}.{name} has no description"


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_the_schema_matches_the_handler_signature(tool):
    """A drifted schema fails at model-call time, which is the worst place to find out."""
    import inspect

    signature = inspect.signature(tool.handler)
    positional = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    }
    required_by_signature = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    assert set(tool.parameters["properties"]) == positional
    assert set(tool.parameters["required"]) == required_by_signature


def test_the_specs_are_json_serializable():
    # They go into an HTTP body; a schema that cannot be dumped fails the whole turn.
    assert json.loads(json.dumps(tool_specs()))


def test_declared_tools_are_what_the_prompt_names():
    from src.agent.prompts import system_prompt

    prompt = system_prompt(config=CONFIG)

    for name in TOOLS_BY_NAME:
        assert name in prompt


# ============================================================== get_market_forecast


def test_get_market_forecast_returns_the_row_and_its_regime():
    conn = FakeWarehouse([FORECAST_ROW], [REGIME_ROW])

    result = get_market_forecast("nvda", config=CONFIG, conn=conn)

    assert result["found"] is True
    assert result["ticker"] == "NVDA"
    assert result["model_used"] == "news_markov"
    assert result["prob_loss_gt_5pct"] == 0.11
    assert result["regime"]["current_news_signal"] == -0.4
    assert result["regime"]["high_vol_sigma"] == 0.034


def test_get_market_forecast_normalizes_the_ticker_before_querying():
    conn = FakeWarehouse([FORECAST_ROW], [REGIME_ROW])

    get_market_forecast("  nvda  ", config=CONFIG, conn=conn)

    assert conn.executed[0].params == {"ticker": "NVDA"}


def test_get_market_forecast_binds_every_value_and_qualifies_every_table():
    conn = FakeWarehouse([FORECAST_ROW], [REGIME_ROW])

    get_market_forecast("NVDA", config=CONFIG, conn=conn)

    forecast, regime = conn.executed
    assert "market_intel.gold.forecast_runs" in forecast.sql
    assert "market_intel.gold.regime_states" in regime.sql
    # The values live in the parameter dict, never in the statement.
    assert ":ticker" in forecast.sql and ":as_of_date" in regime.sql
    assert regime.params == {"ticker": "NVDA", "as_of_date": date(2026, 8, 7)}
    for statement in (forecast.sql, regime.sql):
        assert "NVDA" not in statement
        assert "2026-08-07" not in statement


def test_the_regime_row_is_matched_on_the_forecasts_own_date():
    # Not "today" and not the latest regime row: the pair has to describe the same fit.
    conn = FakeWarehouse([FORECAST_ROW], [REGIME_ROW])

    get_market_forecast("NVDA", config=CONFIG, conn=conn)

    assert conn.executed[1].params["as_of_date"] == FORECAST_ROW["as_of_date"]


def test_dates_and_decimals_come_back_json_safe():
    conn = FakeWarehouse([FORECAST_ROW], [REGIME_ROW])

    result = get_market_forecast("NVDA", config=CONFIG, conn=conn)

    assert result["as_of_date"] == "2026-08-07"
    assert result["generated_at"].startswith("2026-08-10T22:41")
    assert result["current_price"] == 182.5
    # It becomes a tool message, so it has to survive json.dumps without a custom encoder.
    assert json.loads(json.dumps(result))["forecast_id"] == "f-1"


def test_a_ticker_with_no_forecast_is_a_result_not_an_error():
    conn = FakeWarehouse([], [])

    result = get_market_forecast("AMD", config=CONFIG, conn=conn)

    assert result == {
        "found": False,
        "ticker": "AMD",
        "message": result["message"],
    }
    assert "AMD" in result["message"]
    # No point asking for a regime row when there is no forecast to match it to.
    assert len(conn.executed) == 1


def test_a_forecast_without_a_regime_row_still_returns_the_forecast():
    conn = FakeWarehouse([FORECAST_ROW], [])

    result = get_market_forecast("NVDA", config=CONFIG, conn=conn)

    assert result["found"] is True
    assert result["regime"] is None


def test_the_latest_forecast_prefers_the_news_aware_model():
    sql = forecast_sql("market_intel")

    assert "ORDER BY as_of_date DESC" in sql
    assert sql.index("'news_markov' THEN 0") < sql.index("'gbm' THEN 2")
    assert MODEL_PREFERENCE == ("news_markov", "markov", "gbm")
    assert sql.rstrip().endswith("LIMIT 1")


def test_regime_sql_is_keyed_on_the_pair():
    sql = regime_sql("market_intel")

    assert "WHERE ticker = :ticker AND as_of_date = :as_of_date" in sql


@pytest.mark.parametrize("bad", ["", "  ", "tesla stock", "NVDA; DROP TABLE", "12345", None])
def test_a_non_ticker_is_rejected_before_any_query(bad):
    conn = FakeWarehouse([FORECAST_ROW])

    with pytest.raises(ToolError):
        get_market_forecast(bad, config=CONFIG, conn=conn)

    assert conn.executed == []


# =============================================================== search_market_news


def test_search_market_news_returns_the_documented_fields():
    client = FakeIndexClient(rows=[news_row()])

    result = search_market_news("NVDA", "data center demand", config=CONFIG, index_client=client)

    assert result["count"] == 1
    article = result["articles"][0]
    assert set(article) == {
        "title",
        "publisher",
        "published_at",
        "sentiment_label",
        "article_url",
        "snippet",
    }
    assert article["publisher"] == "Reuters"
    assert article["sentiment_label"] == "positive"


def test_the_query_is_hybrid_filtered_by_ticker_against_the_configured_index():
    client = FakeIndexClient(rows=[news_row()])

    search_market_news("nvda", "export licences", k=3, config=CONFIG, index_client=client)

    call = client.calls[0]
    assert call["index_name"] == "market_intel.silver.news_index"
    assert call["query_type"] == "HYBRID"
    assert call["num_results"] == 3
    assert call["query_text"] == "export licences"
    # The filter is a JSON document built by json.dumps, not string concatenation.
    assert json.loads(call["filters_json"]) == {"ticker": "NVDA"}
    assert set(call["columns"]) == set(NEWS_COLUMNS)


def test_the_tool_and_the_setup_script_agree_on_the_index_name():
    # Two modules resolve the same config section; drift here means the agent queries an index
    # that nothing creates.
    index_fqn, top_k = tools._search_settings(CONFIG)

    assert index_fqn == search_settings(CONFIG).index_fqn
    assert top_k == search_settings(CONFIG).top_k


def test_results_are_read_by_column_name_not_position():
    # The index returns columns in whatever order the manifest declares.
    reversed_columns = list(reversed(NEWS_COLUMNS))
    row = news_row()
    client = FakeIndexClient(columns=reversed_columns, rows=[list(reversed(row))])

    result = search_market_news("NVDA", "anything", config=CONFIG, index_client=client)

    assert result["articles"][0]["title"] == "Nvidia beats on data-center revenue"
    assert result["articles"][0]["publisher"] == "Reuters"


def test_the_snippet_drops_the_repeated_title():
    client = FakeIndexClient(rows=[news_row()])

    result = search_market_news("NVDA", "anything", config=CONFIG, index_client=client)

    assert result["articles"][0]["snippet"] == "The company reported..."


def test_a_long_snippet_is_truncated():
    client = FakeIndexClient(rows=[news_row(title="T", embedding_text="T\n" + "x" * 900)])

    snippet = search_market_news("NVDA", "q", config=CONFIG, index_client=client)["articles"][0][
        "snippet"
    ]

    assert len(snippet) <= tools.SNIPPET_CHARS + 3
    assert snippet.endswith("...")


def test_no_matching_news_is_an_answerable_result():
    client = FakeIndexClient(rows=[])

    result = search_market_news("NVDA", "lithium mining", config=CONFIG, index_client=client)

    assert result["count"] == 0
    assert result["articles"] == []
    assert "No indexed news" in result["message"]


def test_k_is_clamped_to_the_documented_range():
    client = FakeIndexClient(rows=[])

    search_market_news("NVDA", "q", k=500, config=CONFIG, index_client=client)
    search_market_news("NVDA", "q", k=0, config=CONFIG, index_client=client)

    assert client.calls[0]["num_results"] == tools.MAX_SEARCH_RESULTS
    assert client.calls[1]["num_results"] == CONFIG["search"]["top_k"]


def test_an_empty_query_is_rejected():
    client = FakeIndexClient()

    with pytest.raises(ToolError):
        search_market_news("NVDA", "   ", config=CONFIG, index_client=client)

    assert client.calls == []


# ================================================================= update_watchlist


def test_update_watchlist_adds_and_returns_the_new_list():
    db = FakeLakebase(["NVDA"])

    result = update_watchlist("add", "amd", lakebase=db)

    assert result["watchlist"] == ["AMD", "NVDA"]
    assert result["changed"] is True
    assert result["action"] == "add"
    assert result["ticker"] == "AMD"
    assert "added" in result["message"]


def test_update_watchlist_removes():
    db = FakeLakebase(["NVDA", "AMD"])

    result = update_watchlist("remove", "AMD", lakebase=db)

    assert result["watchlist"] == ["NVDA"]
    assert result["changed"] is True


def test_adding_a_ticker_already_present_reports_no_change():
    # The agent must confirm what happened, not what was requested.
    db = FakeLakebase(["NVDA"])

    result = update_watchlist("add", "NVDA", lakebase=db)

    assert result["changed"] is False
    assert "already" in result["message"]


def test_removing_an_absent_ticker_reports_no_change():
    db = FakeLakebase(["NVDA"])

    result = update_watchlist("remove", "AMD", lakebase=db)

    assert result["changed"] is False
    assert result["watchlist"] == ["NVDA"]


@pytest.mark.parametrize("action", ["", "delete", "ADD ticker", None])
def test_an_unknown_action_is_rejected(action):
    db = FakeLakebase()

    with pytest.raises(ToolError):
        update_watchlist(action, "NVDA", lakebase=db)

    assert db.calls == []


def test_the_watchlist_write_never_sees_an_unvalidated_symbol():
    db = FakeLakebase()

    with pytest.raises(ToolError):
        update_watchlist("add", "buy tesla", lakebase=db)

    assert db.calls == []


# ============================================================= save_research_report


def test_save_research_report_returns_the_new_id():
    db = FakeLakebase()

    result = save_research_report("nvda", "Why is risk elevated?", "# Answer\n...", lakebase=db)

    assert result["saved"] is True
    assert result["report_id"] == "report-1"
    assert result["ticker"] == "NVDA"
    assert db.reports[0]["question"] == "Why is risk elevated?"


def test_a_report_can_carry_the_forecast_it_quoted():
    db = FakeLakebase()

    save_research_report("NVDA", "q", "body", forecast_id="f-1", lakebase=db)

    assert db.reports[0]["forecast_id"] == "f-1"


@pytest.mark.parametrize("body", ["", "   ", None])
def test_an_empty_report_is_refused(body):
    db = FakeLakebase()

    with pytest.raises(ToolError):
        save_research_report("NVDA", "q", body, lakebase=db)

    assert db.reports == []


# ===================================================================== execute_tool


def test_execute_tool_dispatches_by_name():
    db = FakeLakebase(["NVDA"])

    result = execute_tool("update_watchlist", {"action": "add", "ticker": "AMD"}, lakebase=db)

    assert result["watchlist"] == ["AMD", "NVDA"]


def test_execute_tool_forwards_only_the_context_a_tool_accepts():
    # The agent holds one context for all four tools; handing index_client to a Lakebase write
    # would be a TypeError mid-turn.
    db = FakeLakebase(["NVDA"])

    result = execute_tool(
        "update_watchlist",
        {"action": "remove", "ticker": "NVDA"},
        lakebase=db,
        index_client=FakeIndexClient(),
        conn=None,
        config=CONFIG,
    )

    assert result["watchlist"] == []


def test_execute_tool_rejects_an_unknown_name():
    with pytest.raises(ToolError, match="unknown tool"):
        execute_tool("get_stock_advice", {"ticker": "NVDA"})


def test_execute_tool_reports_a_missing_argument():
    with pytest.raises(ToolError, match="missing required"):
        execute_tool("search_market_news", {"ticker": "NVDA"})


def test_execute_tool_reports_a_hallucinated_argument():
    with pytest.raises(ToolError, match="unexpected argument"):
        execute_tool("get_market_forecast", {"ticker": "NVDA", "horizon": 30})


# =========================================================== the warehouse read path


def test_the_warehouse_connection_is_closed_when_the_helper_opened_it():
    warehouse = FakeWarehouse([{"one": 1}])
    delta.configure(
        delta.WarehouseSettings("host.test", "/sql/1.0/warehouses/w", access_token="t"),
        connect=lambda **kwargs: warehouse,
    )
    try:
        assert delta.query("SELECT 1 AS one") == [{"one": 1}]
    finally:
        delta.configure()

    assert warehouse.closed is True


def test_an_injected_connection_is_left_open():
    warehouse = FakeWarehouse([{"one": 1}])

    delta.query("SELECT 1 AS one", conn=warehouse)

    assert warehouse.closed is False


def test_query_one_returns_none_on_an_empty_result():
    assert delta.query_one("SELECT 1", conn=FakeWarehouse([])) is None


def test_warehouse_settings_accept_an_id_or_a_path():
    from_id = delta.settings_from_env(
        {"DATABRICKS_HOST": "https://dbc-1.cloud.databricks.com/", "DATABRICKS_WAREHOUSE_ID": "abc"}
    )
    from_path = delta.settings_from_env(
        {"DATABRICKS_SERVER_HOSTNAME": "dbc-1.cloud.databricks.com", "DATABRICKS_HTTP_PATH": "/p"}
    )

    assert from_id.server_hostname == "dbc-1.cloud.databricks.com"
    assert from_id.http_path == "/sql/1.0/warehouses/abc"
    assert from_path.http_path == "/p"


def test_incomplete_warehouse_settings_say_what_is_missing():
    with pytest.raises(delta.WarehouseConfigError, match="DATABRICKS_HTTP_PATH"):
        delta.settings_from_env({"DATABRICKS_HOST": "dbc-1.cloud.databricks.com"})


def test_a_token_is_used_when_present_and_never_logged_into_the_query():
    settings = delta.WarehouseSettings("h", "/p", access_token="secret-token")

    kwargs = settings.connect_kwargs()

    assert kwargs["access_token"] == "secret-token"
    assert "credentials_provider" not in kwargs


# ------------------------------------------------------------------- live (opt-in)
# The C-7 integration test. Needs the warehouse, the index and a Lakebase role:
#   AGENT_LIVE_TEST=1 .venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -k live -q


@pytest.mark.skipif(os.environ.get("AGENT_LIVE_TEST") != "1", reason="set AGENT_LIVE_TEST=1")
def test_live_tools_round_trip():
    forecast = get_market_forecast("NVDA")
    assert forecast["found"] in (True, False)
    if forecast["found"]:
        assert forecast["ticker"] == "NVDA"
        assert 0.0 <= forecast["prob_positive"] <= 1.0

    news = search_market_news("NVDA", "data center demand", k=3)
    assert news["count"] <= 3
    for article in news["articles"]:
        assert article["title"]

    added = update_watchlist("add", "AMD")
    assert "AMD" in added["watchlist"]
    report = save_research_report("AMD", "live test", "# live test\nbody")
    assert report["report_id"]
    removed = update_watchlist("remove", "AMD")
    assert "AMD" not in removed["watchlist"]
