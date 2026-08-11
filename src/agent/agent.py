"""The agent loop (spec C-4).

A plain tool-calling loop, capped at roughly 6 tool iterations:

1. Send the system prompt, the conversation history and the tool schemas via
   ``call_model("agent", ...)``.
2. Execute whatever tool calls come back.
3. Append the tool results to the history.
4. Stop when the model returns a final text answer, or when the iteration cap is hit.

A typical turn for "why is downside risk elevated this week?" is
``get_market_forecast`` -> ``search_market_news`` -> a written explanation that cites both
the Gold numbers and the retrieved article titles, and invents neither.

No routing, no escalation, no sub-agents.

WHAT THE CAP IS FOR. Six iterations is not a performance budget; it is the guard against a model
that keeps calling tools instead of answering — usually by re-asking for the same forecast in a
loop. When it trips, the turn ends with a truthful "I could not finish" rather than a fabricated
answer, and :attr:`AgentResult.hit_iteration_limit` says so, so the app can show it.

A FAILING TOOL IS A RESULT, NOT AN EXCEPTION. Every tool error — an unknown name, an invalid
ticker, malformed JSON arguments, a database that is down — is serialized back to the model as
the tool's result. The model can then say what went wrong or try a different argument, which is
strictly better than the alternative of the turn dying with a traceback. A failure of the MODEL
call itself does propagate: there is nothing left to recover with.

Telemetry is recorded by ``call_model`` on every call, success or failure, so the loop does not
record anything itself (C-3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent import tools as tool_module
from src.agent.prompts import system_prompt
from src.llm.call_model import ModelResponse, ToolCall, call_model

log = logging.getLogger(__name__)

__all__ = [
    "MAX_TOOL_ITERATIONS",
    "AgentResult",
    "ToolInvocation",
    "run_agent",
]

#: The spec's cap: at most six model turns that end in tool calls, then one final answer.
MAX_TOOL_ITERATIONS = 6

#: The task name ``call_model`` resolves to ``model.agent_endpoint``. There is exactly one.
TASK = "agent"

ITERATION_LIMIT_MESSAGE = (
    "I could not finish this within the tool-call limit, so I am stopping rather than "
    "guessing at an answer. Please narrow the question and try again."
)


@dataclass(frozen=True)
class ToolInvocation:
    """One executed tool call, kept for the UI's "what did it do" panel and for the tests."""

    name: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    ok: bool


@dataclass
class AgentResult:
    """The outcome of one user turn."""

    text: str
    messages: list[dict] = field(default_factory=list)
    """The full conversation including this turn: system, user, assistant and tool messages. Pass
    it back as ``history`` to continue the conversation."""

    tool_calls: list[ToolInvocation] = field(default_factory=list)
    responses: list[ModelResponse] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_limit: bool = False

    @property
    def in_tokens(self) -> int:
        return sum(response.in_tokens or 0 for response in self.responses)

    @property
    def out_tokens(self) -> int:
        return sum(response.out_tokens or 0 for response in self.responses)

    @property
    def latency_ms(self) -> float:
        return sum(response.latency_ms for response in self.responses)


def _tool_result_message(call: ToolCall, payload: Mapping[str, Any]) -> dict:
    """One ``role: tool`` message, which is how a result gets back to the model.

    ``default=str`` on the dump is a backstop, not a licence: the tools already return
    JSON-safe dicts, but a stringified value beats an exception thrown mid-turn.
    """
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(payload, default=str),
    }


def _execute(
    call: ToolCall,
    context: Mapping[str, Any],
    execute: Callable[..., dict],
) -> ToolInvocation:
    """Run one tool call, converting any failure into a result the model can read."""
    try:
        arguments = call.parse_arguments()
    except Exception as exc:
        log.warning("tool arguments were not usable tool=%s error=%s", call.name, exc)
        return ToolInvocation(call.name, {}, {"error": str(exc)}, ok=False)

    try:
        result = execute(call.name, arguments, **context)
    except Exception as exc:
        # Deliberately broad: a tool talks to a warehouse, a search index and Postgres, and the
        # loop's job is to keep the turn alive and tell the model what happened.
        log.warning("tool failed tool=%s error=%s: %s", call.name, type(exc).__name__, exc)
        return ToolInvocation(
            call.name,
            arguments,
            {"error": f"{type(exc).__name__}: {exc}", "tool": call.name},
            ok=False,
        )

    log.info("tool ok tool=%s args=%s", call.name, sorted(arguments))
    return ToolInvocation(call.name, arguments, result, ok=True)


def run_agent(
    question: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    specs: Sequence[Mapping[str, Any]] | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    model: Callable[..., ModelResponse] = call_model,
    execute: Callable[..., dict] = tool_module.execute_tool,
    **model_kwargs: Any,
) -> AgentResult:
    """Answer one question, calling tools as the model asks for them.

    ``history`` continues a conversation: pass a previous :attr:`AgentResult.messages` back in.
    The system prompt is prepended only when the history does not already start with one, so
    continuing a conversation does not stack duplicate instructions.

    ``context`` holds the injectables the tools need (``conn``, ``index_client``, ``lakebase``,
    ``config``) — the app supplies real ones, the tests supply fakes. ``model`` and ``execute``
    are injectable for the same reason.
    """
    messages: list[dict] = [dict(message) for message in (history or ())]
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_prompt(config=config)})
    messages.append({"role": "user", "content": str(question)})

    tool_context = {"config": config, **dict(context or {})}
    declarations = list(specs) if specs is not None else tool_module.tool_specs()

    result = AgentResult(text="", messages=messages)

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response = model(TASK, messages, declarations, config=config, **model_kwargs)
        result.responses.append(response)
        # The raw assistant message verbatim when there is one: it carries the tool_calls array
        # the endpoint expects to see echoed back, and re-serializing it would drop fields.
        messages.append(
            dict(response.message)
            if response.message
            else {"role": "assistant", "content": response.text}
        )

        if not response.has_tool_calls:
            result.text = response.text
            return result

        for call in response.tool_calls:
            invocation = _execute(call, tool_context, execute)
            result.tool_calls.append(invocation)
            messages.append(_tool_result_message(call, invocation.result))

    # The cap was reached with the model still asking for tools. Say so rather than presenting
    # the last partial text as if it were an answer.
    log.warning("agent hit the tool-call limit iterations=%d", max_iterations)
    result.hit_iteration_limit = True
    result.text = ITERATION_LIMIT_MESSAGE
    messages.append({"role": "assistant", "content": result.text})
    return result
