"""call_model tests (spec C-3). Mocked HTTP only — no workspace, no network, no real sleeping.

Covers the three cases the checkpoint asks for — a plain completion, a tool-call completion, and
the error paths — plus the two things that are easy to get wrong and impossible to notice later:

- the endpoint comes from config BY TASK, and an unknown task fails instead of defaulting
- every call, successful or not, leaves exactly one telemetry record

The transport is injected (``session``), as are the credentials, the clock and ``sleep``, so
"it backed off once" is asserted rather than endured.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import pytest
import requests

from src.llm import ConfigError, telemetry
from src.llm.call_model import (
    BACKOFF_BASE_SECONDS,
    MAX_ATTEMPTS,
    Credentials,
    ModelAuthError,
    ModelError,
    ModelHTTPError,
    ModelTransportError,
    call_model,
    resolve_endpoint,
    workspace_credentials,
)

AGENT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
SLM_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"
CONFIG = {"model": {"agent_endpoint": AGENT_ENDPOINT, "slm_endpoint": SLM_ENDPOINT}}

HOST = "https://dbc-47c4ef4d-9564.cloud.databricks.com"
TOKEN = "dapi-NEVER-LOG-ME-0123456789"
CREDS = Credentials(host=HOST, headers={"Authorization": f"Bearer {TOKEN}"})

MESSAGES = [
    {"role": "system", "content": "You explain market forecasts."},
    {"role": "user", "content": "Why is downside risk elevated for NVDA this week?"},
]

FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_market_forecast",
        "description": "Latest gold forecast row for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}


# --------------------------------------------------------------------------- fakes


class FakeClock:
    """Monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class FakeResponse:
    status_code: int
    payload: object | None = None
    json_raises: bool = False

    def json(self) -> object:
        if self.json_raises:
            raise ValueError("not json")
        return self.payload


@dataclass
class RecordedPost:
    url: str
    body: dict
    headers: dict
    timeout: float | None


@dataclass
class FakeSession:
    """Returns queued responses (or raises queued exceptions) and records every call."""

    responses: list
    calls: list[RecordedPost] = field(default_factory=list)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            RecordedPost(url=url, body=dict(json or {}), headers=dict(headers or {}), timeout=timeout)
        )
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def completion(
    *,
    content: str | None = "Downside risk is elevated because...",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str = "meta-llama-3.3-70b-instruct",
    usage: dict | None = None,
) -> FakeResponse:
    """An OpenAI-compatible chat completion, the shape the Foundation Model API returns."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return FakeResponse(
        status_code=200,
        payload={
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage if usage is not None else {
                "prompt_tokens": 412,
                "completion_tokens": 137,
                "total_tokens": 549,
            },
        },
    )


def invoke(responses, *, task="agent", clock=None, **kwargs):
    clock = clock or FakeClock()
    session = FakeSession(responses=list(responses))
    response = call_model(
        task,
        kwargs.pop("messages", MESSAGES),
        config=CONFIG,
        session=session,
        credentials=CREDS,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )
    return response, session, clock


@pytest.fixture(autouse=True)
def quiet_telemetry():
    """Buffer telemetry in memory for the duration of each test, then reset the module."""
    telemetry.configure(telemetry.MODE_LOG)
    telemetry.clear()
    yield
    telemetry.clear()
    telemetry.configure(None)


# ---------------------------------------------------------------- endpoint resolution


def test_endpoint_is_resolved_from_config_by_task():
    assert resolve_endpoint("agent", CONFIG) == AGENT_ENDPOINT
    assert resolve_endpoint("slm", CONFIG) == SLM_ENDPOINT


def test_unknown_task_raises_instead_of_defaulting():
    # A typo'd task silently falling through to the agent endpoint is the failure this prevents.
    with pytest.raises(ValueError, match="unknown task"):
        resolve_endpoint("Agent", CONFIG)


def test_missing_endpoint_key_raises_a_config_error():
    with pytest.raises(ConfigError, match="model.slm_endpoint"):
        resolve_endpoint("slm", {"model": {"agent_endpoint": AGENT_ENDPOINT}})


def test_the_repository_config_defines_both_endpoints():
    from src.llm import load_config

    config = load_config()
    assert resolve_endpoint("agent", config)
    assert resolve_endpoint("slm", config)


# ------------------------------------------------------------------------- success


def test_successful_call_posts_to_the_task_endpoint_and_parses_the_completion():
    response, session, _ = invoke([completion()])

    call = session.calls[0]
    assert call.url == f"{HOST}/serving-endpoints/{AGENT_ENDPOINT}/invocations"
    assert call.body["messages"] == MESSAGES
    assert "tools" not in call.body  # nothing invented when the caller passed none
    assert "response_format" not in call.body
    assert call.headers["Authorization"] == f"Bearer {TOKEN}"

    assert response.task == "agent"
    assert response.endpoint == AGENT_ENDPOINT
    assert response.model == "meta-llama-3.3-70b-instruct"
    assert response.text.startswith("Downside risk is elevated")
    assert response.tool_calls == ()
    assert response.has_tool_calls is False
    assert (response.in_tokens, response.out_tokens) == (412, 137)


def test_slm_task_uses_the_slm_endpoint():
    _, session, _ = invoke([completion()], task="slm")

    assert session.calls[0].url.endswith(f"/serving-endpoints/{SLM_ENDPOINT}/invocations")


def test_tools_and_response_format_are_passed_through_unchanged():
    schema = {"type": "json_schema", "json_schema": {"name": "event", "schema": {"type": "object"}}}

    _, session, _ = invoke([completion()], tools=[FORECAST_TOOL], response_format=schema)

    body = session.calls[0].body
    assert body["tools"] == [FORECAST_TOOL]
    assert body["response_format"] == schema


def test_missing_usage_leaves_token_counts_none():
    response, _, _ = invoke([completion(usage={})])

    # NULL means "the endpoint did not report it", which is not the same claim as zero.
    assert response.in_tokens is None
    assert response.out_tokens is None


# ----------------------------------------------------------------------- tool calls


def test_tool_call_response_is_parsed_and_the_raw_message_is_preserved():
    raw_call = {
        "id": "call_9f3",
        "type": "function",
        "function": {"name": "get_market_forecast", "arguments": '{"ticker": "NVDA"}'},
    }
    response, _, _ = invoke(
        [completion(content=None, tool_calls=[raw_call], finish_reason="tool_calls")],
        tools=[FORECAST_TOOL],
    )

    assert response.has_tool_calls
    assert response.finish_reason == "tool_calls"
    tool_call = response.tool_calls[0]
    assert (tool_call.id, tool_call.name) == ("call_9f3", "get_market_forecast")
    assert tool_call.parse_arguments() == {"ticker": "NVDA"}
    assert response.text == ""  # a tool-call turn has no prose

    # The agent loop appends this message verbatim to the conversation, so it must survive intact.
    assert response.message["tool_calls"] == [raw_call]


def test_malformed_tool_arguments_raise_a_named_model_error():
    raw_call = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "update_watchlist", "arguments": "{'ticker': NVDA"},
    }
    response, _, _ = invoke([completion(content=None, tool_calls=[raw_call])])

    with pytest.raises(ModelError, match="update_watchlist"):
        response.tool_calls[0].parse_arguments()


def test_tool_calls_without_a_name_are_ignored():
    response, _, _ = invoke([completion(content=None, tool_calls=[{"id": "x", "function": {}}])])

    assert response.tool_calls == ()


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_raise_immediately_without_retrying(status):
    payload = {"error_code": "PERMISSION_DENIED", "message": "does not have CAN QUERY"}
    with pytest.raises(ModelAuthError) as excinfo:
        invoke([FakeResponse(status, payload=payload)])

    assert str(status) in str(excinfo.value)
    assert "CAN QUERY" in str(excinfo.value)


def test_5xx_retries_to_max_attempts_then_raises():
    responses = [FakeResponse(503, payload={"message": "endpoint is starting"})] * MAX_ATTEMPTS
    clock = FakeClock()

    with pytest.raises(ModelHTTPError) as excinfo:
        invoke(responses, clock=clock)

    assert len(clock.sleeps) == MAX_ATTEMPTS - 1
    assert clock.sleeps == sorted(clock.sleeps)  # exponential, not constant
    assert BACKOFF_BASE_SECONDS <= clock.sleeps[0] <= BACKOFF_BASE_SECONDS * 1.25
    assert AGENT_ENDPOINT in str(excinfo.value)


def test_429_then_success_backs_off_once():
    response, session, clock = invoke([FakeResponse(429, payload={"message": "rate limited"}), completion()])

    assert response.text
    assert len(session.calls) == 2
    assert len(clock.sleeps) == 1


def test_transport_failure_retries_then_raises_without_the_host_or_token():
    leaky = requests.ConnectionError(f"failed to reach {HOST}/serving-endpoints with {TOKEN}")

    with pytest.raises(ModelTransportError) as excinfo:
        invoke([leaky] * MAX_ATTEMPTS)

    message = str(excinfo.value)
    assert TOKEN not in message
    assert excinfo.value.__cause__ is None


def test_a_200_with_no_choices_is_an_error_not_an_empty_answer():
    with pytest.raises(ModelHTTPError, match="no choices"):
        invoke([FakeResponse(200, payload={"model": "x", "choices": []})])


def test_error_body_without_json_still_raises():
    responses = [FakeResponse(500, json_raises=True)] * MAX_ATTEMPTS

    with pytest.raises(ModelHTTPError, match="no JSON body"):
        invoke(responses)


@pytest.mark.parametrize(
    "messages",
    [[], [{"content": "no role here"}], ["not a mapping"]],
)
def test_malformed_messages_are_rejected_before_any_request(messages):
    with pytest.raises(ValueError):
        invoke([completion()], messages=messages)


def test_nothing_credential_bearing_reaches_the_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger="src.llm.call_model"):
        with pytest.raises(ModelAuthError):
            invoke([FakeResponse(403, payload={"message": "denied"})])
        invoke([completion()])

    logged = caplog.text
    assert "status=403" in logged
    assert AGENT_ENDPOINT in logged
    for forbidden in (TOKEN, "Bearer ", "Why is downside risk elevated"):
        assert forbidden not in logged, f"leaked into logs: {forbidden!r}"


# ------------------------------------------------------------------------ telemetry


def test_a_successful_call_records_one_telemetry_row():
    invoke([completion()])

    (record,) = telemetry.buffered()
    assert record.task == "agent"
    assert record.model == "meta-llama-3.3-70b-instruct"
    assert record.ok is True
    assert (record.in_tokens, record.out_tokens) == (412, 137)
    assert record.latency_ms >= 0.0


def test_a_failed_call_records_a_row_too():
    with pytest.raises(ModelAuthError):
        invoke([FakeResponse(401, payload={"message": "denied"})])

    (record,) = telemetry.buffered()
    assert record.ok is False
    assert record.model == AGENT_ENDPOINT  # no response, so the endpoint name stands in
    assert record.in_tokens is None


def test_a_telemetry_failure_never_fails_the_model_call(monkeypatch):
    def explode(**_kwargs):
        raise RuntimeError("telemetry is down")

    monkeypatch.setattr(telemetry, "record", explode)

    response, _, _ = invoke([completion()])

    assert response.text  # the call still succeeded


# ---------------------------------------------------------------------- credentials


def test_workspace_credentials_prefer_the_environment(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dbc-test.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", TOKEN)

    creds = workspace_credentials()

    # A host without a scheme still has to produce a usable URL.
    assert creds.host == "https://dbc-test.cloud.databricks.com"
    assert creds.headers["Authorization"] == f"Bearer {TOKEN}"


# ----------------------------------------------------------------- live (opt-in)
# Not part of the default run: it needs a workspace and it costs a real inference. This is the
# C-d verify-first item "Foundation Model endpoint responds to one call", kept as code so it can
# be re-run after a config change instead of remembered as a one-off.
#   LLM_LIVE_TEST=1 .venv/Scripts/python.exe -m pytest tests/test_call_model.py -k live -q


@pytest.mark.skipif(os.environ.get("LLM_LIVE_TEST") != "1", reason="set LLM_LIVE_TEST=1 to run")
def test_live_agent_endpoint_answers_one_call():
    response = call_model(
        "agent",
        [{"role": "user", "content": "Reply with the single word: ready"}],
    )

    assert response.text.strip()
    assert response.model
    assert telemetry.buffered()[-1].ok is True
