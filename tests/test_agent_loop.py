"""Agent loop and system prompt tests (spec C-4).

The loop is scripted rather than mocked loosely: a fake model returns a queued sequence of
completions, so "one tool-call turn, then a text turn" is asserted turn by turn — which messages
were sent, which tools ran, what came back.

Three properties matter more than the happy path:

- THE CAP HOLDS. A model that keeps calling tools must be stopped at six iterations with an
  honest "I could not finish", not left to spend tokens or to emit a fabricated answer.
- A FAILING TOOL DOES NOT END THE TURN. Tool errors come back to the model as results, because a
  traceback out of the loop is a dead chat window and the model can often recover.
- TELEMETRY IS RECORDED ON EVERY MODEL CALL. Asserted through the real ``call_model`` over a fake
  HTTP session rather than by trusting the wiring, since the loop itself records nothing (C-3).
"""

from __future__ import annotations

import json

import pytest

from src.agent.agent import (
    ITERATION_LIMIT_MESSAGE,
    MAX_TOOL_ITERATIONS,
    AgentResult,
    run_agent,
)
from src.agent.prompts import SYSTEM_PROMPT_TEMPLATE, system_prompt
from src.llm import telemetry
from src.llm.call_model import ModelResponse, ToolCall, call_model

CONFIG = {
    "catalog": "market_intel",
    "news": {"half_life_days": 2},
    "forecast": {"horizon_days": 5},
    "model": {"agent_endpoint": "databricks-meta-llama-3-3-70b-instruct"},
}


# --------------------------------------------------------------------------- fakes


class ScriptedModel:
    """Returns queued responses and records the messages and tools it was handed."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, task, messages, tools=None, response_format=None, **kwargs):
        # Snapshot: the loop mutates the same list in place.
        self.calls.append(
            {
                "task": task,
                "messages": [dict(message) for message in messages],
                "tools": list(tools or ()),
            }
        )
        if not self.responses:
            raise AssertionError("the loop asked for more turns than the script provides")
        return self.responses.pop(0)


def text_turn(content="Downside risk is elevated because the model is in the turbulent regime."):
    return ModelResponse(
        task="agent",
        endpoint="e",
        model="m",
        text=content,
        tool_calls=(),
        finish_reason="stop",
        in_tokens=100,
        out_tokens=20,
        latency_ms=42.0,
        message={"role": "assistant", "content": content},
    )


def tool_turn(*calls):
    """A completion whose assistant message asks for tools. ``calls`` are (id, name, args)."""
    tool_calls = tuple(ToolCall(id=cid, name=name, arguments=args) for cid, name, args in calls)
    raw = [
        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        for call in tool_calls
    ]
    return ModelResponse(
        task="agent",
        endpoint="e",
        model="m",
        text="",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        in_tokens=90,
        out_tokens=30,
        latency_ms=17.0,
        message={"role": "assistant", "content": None, "tool_calls": raw},
    )


class RecordingExecutor:
    """Stands in for ``tools.execute_tool``."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {"found": True, "ticker": "NVDA"}
        self.raises = raises
        self.calls: list[tuple] = []

    def __call__(self, name, arguments=None, **context):
        self.calls.append((name, dict(arguments or {}), context))
        if self.raises is not None:
            raise self.raises
        return self.result


def always_tools(n=MAX_TOOL_ITERATIONS + 4):
    return ScriptedModel(
        *[tool_turn((f"c{i}", "get_market_forecast", '{"ticker": "NVDA"}')) for i in range(n)]
    )


# ==================================================================== the happy path


def test_one_tool_call_turn_then_a_text_turn():
    model = ScriptedModel(
        tool_turn(("call-1", "get_market_forecast", '{"ticker": "NVDA"}')),
        text_turn(),
    )
    execute = RecordingExecutor()

    result = run_agent("Why is NVDA risky?", config=CONFIG, model=model, execute=execute)

    assert result.text.startswith("Downside risk is elevated")
    assert result.iterations == 2
    assert result.hit_iteration_limit is False
    assert [call.name for call in result.tool_calls] == ["get_market_forecast"]
    assert execute.calls[0][1] == {"ticker": "NVDA"}


def test_the_tool_result_goes_back_as_a_tool_message():
    model = ScriptedModel(
        tool_turn(("call-1", "get_market_forecast", '{"ticker": "NVDA"}')),
        text_turn(),
    )
    execute = RecordingExecutor(result={"found": True, "prob_positive": 0.52})

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call-1"
    assert json.loads(tool_messages[0]["content"])["prob_positive"] == 0.52
    # The second model call sees the result.
    assert model.calls[1]["messages"][-1]["role"] == "tool"


def test_the_assistant_message_is_appended_verbatim():
    # It carries the tool_calls array the endpoint expects echoed back on the next request.
    model = ScriptedModel(
        tool_turn(("call-1", "get_market_forecast", '{"ticker": "NVDA"}')),
        text_turn(),
    )

    result = run_agent("q", config=CONFIG, model=model, execute=RecordingExecutor())

    assistant = [m for m in result.messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "get_market_forecast"


def test_a_text_first_answer_calls_no_tools():
    model = ScriptedModel(text_turn("No forecast is available for that ticker."))
    execute = RecordingExecutor()

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    assert result.iterations == 1
    assert execute.calls == []
    assert result.tool_calls == []


def test_several_tool_calls_in_one_turn_all_run_in_order():
    model = ScriptedModel(
        tool_turn(
            ("c1", "get_market_forecast", '{"ticker": "NVDA"}'),
            ("c2", "search_market_news", '{"ticker": "NVDA", "query": "demand"}'),
        ),
        text_turn(),
    )
    execute = RecordingExecutor()

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    assert [name for name, _, _ in execute.calls] == ["get_market_forecast", "search_market_news"]
    assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["c1", "c2"]
    assert result.iterations == 2


# ================================================================ prompt and history


def test_the_system_prompt_leads_the_conversation():
    model = ScriptedModel(text_turn())

    result = run_agent("q", config=CONFIG, model=model)

    assert result.messages[0]["role"] == "system"
    assert "market research explainer" in result.messages[0]["content"]
    assert result.messages[1] == {"role": "user", "content": "q"}


def test_continuing_a_conversation_does_not_stack_system_prompts():
    first = run_agent("q1", config=CONFIG, model=ScriptedModel(text_turn()))

    second = run_agent(
        "q2", first.messages, config=CONFIG, model=ScriptedModel(text_turn("second"))
    )

    assert [m["role"] for m in second.messages].count("system") == 1
    assert second.messages[-2]["content"] == "q2"


def test_the_history_passed_in_is_not_mutated():
    first = run_agent("q1", config=CONFIG, model=ScriptedModel(text_turn()))
    before = len(first.messages)

    run_agent("q2", first.messages, config=CONFIG, model=ScriptedModel(text_turn()))

    assert len(first.messages) == before


def test_the_model_receives_the_four_tool_schemas():
    model = ScriptedModel(text_turn())

    run_agent("q", config=CONFIG, model=model)

    names = {tool["function"]["name"] for tool in model.calls[0]["tools"]}
    assert names == {
        "get_market_forecast",
        "search_market_news",
        "update_watchlist",
        "save_research_report",
    }
    assert model.calls[0]["task"] == "agent"


def test_the_tool_context_reaches_the_executor():
    model = ScriptedModel(
        tool_turn(("c1", "get_market_forecast", '{"ticker": "NVDA"}')), text_turn()
    )
    execute = RecordingExecutor()
    sentinel = object()

    run_agent("q", config=CONFIG, context={"conn": sentinel}, model=model, execute=execute)

    assert execute.calls[0][2]["conn"] is sentinel
    assert execute.calls[0][2]["config"] == CONFIG


# ====================================================================== the loop cap


def test_the_default_cap_is_six_iterations():
    assert MAX_TOOL_ITERATIONS == 6


def test_a_model_that_never_stops_calling_tools_is_cut_off_at_six():
    model = always_tools()
    execute = RecordingExecutor()

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    assert result.iterations == MAX_TOOL_ITERATIONS
    assert len(model.calls) == MAX_TOOL_ITERATIONS
    assert len(execute.calls) == MAX_TOOL_ITERATIONS
    assert result.hit_iteration_limit is True


def test_hitting_the_cap_says_so_instead_of_answering():
    # The failure mode this prevents is a confident-looking answer assembled from nothing.
    result = run_agent("q", config=CONFIG, model=always_tools(), execute=RecordingExecutor())

    assert result.text == ITERATION_LIMIT_MESSAGE
    assert result.messages[-1] == {"role": "assistant", "content": ITERATION_LIMIT_MESSAGE}


def test_the_cap_is_adjustable_for_callers_that_need_a_shorter_leash():
    result = run_agent(
        "q", config=CONFIG, model=always_tools(), execute=RecordingExecutor(), max_iterations=2
    )

    assert result.iterations == 2
    assert result.hit_iteration_limit is True


# ================================================================== failing tools


def test_a_tool_exception_becomes_a_result_the_model_can_read():
    model = ScriptedModel(
        tool_turn(("c1", "get_market_forecast", '{"ticker": "not a ticker"}')),
        text_turn("That is not a ticker symbol I can look up."),
    )
    execute = RecordingExecutor(raises=ValueError("'not a ticker' is not a ticker symbol"))

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    assert result.tool_calls[0].ok is False
    assert "not a ticker" in result.tool_calls[0].result["error"]
    assert result.text.startswith("That is not a ticker")


def test_malformed_tool_arguments_do_not_raise_out_of_the_loop():
    model = ScriptedModel(
        tool_turn(("c1", "get_market_forecast", "{ticker: NVDA")),
        text_turn("Let me try that again."),
    )
    execute = RecordingExecutor()

    result = run_agent("q", config=CONFIG, model=model, execute=execute)

    assert execute.calls == []  # never dispatched
    assert result.tool_calls[0].ok is False
    assert "not JSON" in result.tool_calls[0].result["error"]


def test_a_model_failure_does_propagate():
    # There is nothing to recover with, and a swallowed model error would look like a bad answer.
    class Broken:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError, match="endpoint down"):
        run_agent("q", config=CONFIG, model=Broken())


# ======================================================================== telemetry


class FakeHTTPResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Two queued completions: a tool-call turn, then a text turn."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        return FakeHTTPResponse(self.payloads.pop(0))


def completion_payload(message, finish_reason="stop"):
    return {
        "model": "meta-llama-3.3-70b-instruct",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 400, "completion_tokens": 60},
    }


def test_every_model_call_in_a_turn_leaves_a_telemetry_record():
    from src.llm.call_model import Credentials

    session = FakeSession(
        [
            completion_payload(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_market_forecast",
                                "arguments": '{"ticker": "NVDA"}',
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
            completion_payload({"role": "assistant", "content": "Here is what the forecast says."}),
        ]
    )
    credentials = Credentials(host="https://workspace.test", headers={"Authorization": "Bearer x"})

    telemetry.configure(telemetry.MODE_LOG)
    telemetry.clear()
    try:
        result = run_agent(
            "q",
            config=CONFIG,
            model=call_model,
            execute=RecordingExecutor(),
            session=session,
            credentials=credentials,
        )
        records = telemetry.buffered()
    finally:
        telemetry.clear()
        telemetry.configure(None)

    assert session.calls == 2
    assert result.iterations == 2
    assert len(records) == 2
    assert {record.task for record in records} == {"agent"}
    assert all(record.ok for record in records)


def test_the_result_totals_the_tokens_it_was_charged_for():
    result = AgentResult(text="x", responses=[text_turn(), text_turn()])

    assert result.in_tokens == 200
    assert result.out_tokens == 40
    assert result.latency_ms == 84.0


# =================================================================== system prompt


def test_the_prompt_states_every_rule_the_spec_requires():
    prompt = system_prompt(config=CONFIG)

    assert "market research explainer" in prompt
    assert "CALL get_market_forecast BEFORE MAKING ANY QUANTITATIVE CLAIM" in prompt
    assert "NEVER INVENT A NUMBER" in prompt
    assert "AND SO DO YOU" in prompt
    assert "NAME THE TITLES" in prompt
    assert "NEVER GIVE INVESTMENT ADVICE" in prompt
    assert "CONFIRM EVERY WRITE" in prompt


@pytest.mark.parametrize("banned", ["buy", "sell", "hold", "price target"])
def test_the_prompt_names_the_advice_it_forbids(banned):
    assert banned in system_prompt(config=CONFIG).lower()


def test_the_decay_assumption_carries_the_configured_half_life():
    # Typed into the prompt, it would drift the day config changes.
    assert "2-trading-day half-life" in system_prompt(config=CONFIG)
    assert "5-trading-day half-life" in system_prompt(config={"news": {"half_life_days": 5}})
    assert "{half_life_days}" in SYSTEM_PROMPT_TEMPLATE


def test_a_fractional_half_life_still_reads_naturally():
    assert "1.5-trading-day" in system_prompt(1.5)


def test_the_prompt_falls_back_when_the_config_has_no_half_life():
    assert "2-trading-day half-life" in system_prompt(config={"catalog": "market_intel"})


class TestHorizonRule:
    """The rule added after a live session found the gap between "no numbers" and "no advice".

    Asked about a MONTH, the agent invented no figure and gave no advice — and then explained how
    to extrapolate the 5-day returns to get one. Methodology was covered by neither rule.
    """

    def test_the_horizon_comes_from_config_not_from_the_template(self):
        # Typed into the prompt, it would drift the day forecast.horizon_days changes — the same
        # reason the half-life is interpolated, and the same silent failure.
        assert "{horizon_days}" in SYSTEM_PROMPT_TEMPLATE
        assert "5-TRADING-DAY HORIZON" in system_prompt(config=CONFIG)
        assert "10-TRADING-DAY HORIZON" in system_prompt(
            config={**CONFIG, "forecast": {"horizon_days": 10}}
        )

    def test_every_horizon_mention_moves_together(self):
        """One typed-in "5-day" left behind would contradict the rule in the same prompt."""
        prompt = system_prompt(config={**CONFIG, "forecast": {"horizon_days": 10}})

        assert "5-day" not in prompt
        assert "5-trading-day" not in prompt.lower()
        assert "10-day horizon" in prompt

    def test_it_falls_back_when_the_config_has_no_horizon(self):
        """A prompt silent on scope is exactly the failure this rule exists for."""
        assert "5-TRADING-DAY HORIZON" in system_prompt(config={"catalog": "market_intel"})

    def test_the_forbidden_methods_are_named_one_by_one(self):
        """"Do not extrapolate" alone leaves annualizing and compounding looking permitted."""
        prompt = system_prompt(config=CONFIG).lower()

        for method in ("extrapolat", "scaling", "annualiz", "compounding"):
            assert method in prompt

    def test_it_closes_the_offer_to_let_the_user_do_the_arithmetic(self):
        """The exact evasion observed: no number written, a recipe for one handed over."""
        # Unwrapped, so re-flowing the paragraph cannot fail this on a line break.
        prompt = " ".join(system_prompt(config=CONFIG).split())

        assert "something the user could do themselves" in prompt
        assert "A method you suggest is a number you caused" in prompt
        assert "describing how to manufacture a figure the system did not produce" in prompt

    def test_the_rule_answers_the_question_rather_than_refusing_it(self):
        """Declining the horizon is the point; declining to explain anything is not."""
        prompt = " ".join(system_prompt(config=CONFIG).split())

        assert "then answer what the 5-day data does show" in prompt

    def test_it_sits_with_the_other_data_scope_rule(self):
        """Rule 3 is rule 2's family — a reader who stops after "never invent a number" has it."""
        prompt = system_prompt(config=CONFIG)

        assert prompt.index("NEVER INVENT A NUMBER") < prompt.index("AND SO DO YOU")
        assert prompt.index("AND SO DO YOU") < prompt.index("GROUND EVERY NEWS CLAIM")

    def test_the_rules_are_numbered_without_a_gap(self):
        """Inserting a rule mid-list is where a renumber gets forgotten."""
        numbers = [
            int(line.split(".", 1)[0])
            for line in system_prompt(config=CONFIG).splitlines()
            if line[:1].isdigit() and ". " in line[:4]
        ]
        assert numbers == list(range(1, len(numbers) + 1))
