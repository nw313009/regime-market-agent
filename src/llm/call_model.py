"""The single model-access abstraction (spec C-3).

Contract::

    def call_model(task: str, messages, tools=None, response_format=None) -> Response

Reads the endpoint name from config by task: ``"agent"`` -> ``model.agent_endpoint`` now,
``"slm"`` -> ``model.slm_endpoint`` later if the stretch goal happens. Wraps the Databricks
Foundation Model API, which is OpenAI-compatible chat completions.

No routing logic. No tiers. Endpoint names must not be scattered as literals through the
repository — that is the entire reason this indirection exists.

TRANSPORT. A plain ``POST {host}/serving-endpoints/{endpoint}/invocations`` with ``requests``,
which is already a dependency, rather than ``databricks-sdk``'s ``serving_endpoints.query``: the
SDK call takes typed ``ChatMessage`` objects and would mean translating the OpenAI-shaped
``messages``, ``tools`` and ``response_format`` this function is handed into SDK dataclasses and
back again. The SDK is still what resolves the CREDENTIAL, so notebooks, jobs and the app all
authenticate as whatever identity they already run as.

Body and response are OpenAI-compatible and are passed through unchanged, so the agent loop
(C-4) can hand back an assistant message with ``tool_calls`` verbatim.

SECURITY. The bearer token never reaches a log record, and neither does a request or response
body: prompts and completions carry user text, and this module logs the endpoint, the status,
the latency and the token counts only.

Every call — including a failed one — appends one record through :mod:`src.llm.telemetry`.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from src.llm import ConfigError, config_section
from src.llm import telemetry as telemetry_module

__all__ = [
    "MAX_ATTEMPTS",
    "TASK_ENDPOINT_KEYS",
    "Credentials",
    "ModelAuthError",
    "ModelError",
    "ModelHTTPError",
    "ModelResponse",
    "ModelTransportError",
    "ToolCall",
    "call_model",
    "resolve_endpoint",
    "workspace_credentials",
]

log = logging.getLogger(__name__)

#: task -> the ``model.*`` config key holding its endpoint name. The whole routing story.
TASK_ENDPOINT_KEYS: Mapping[str, str] = {
    "agent": "agent_endpoint",
    "slm": "slm_endpoint",
}

DEFAULT_TIMEOUT_SECONDS = 120

# Retries exist for the two transient failures a serving endpoint actually produces — a 429 under
# concurrency and a 503 while a scale-to-zero endpoint wakes up. Fewer attempts than the ingestion
# client's five: a model call sits on the interactive path behind a user watching a chat box.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ModelError(RuntimeError):
    """Base class for every model-call failure."""


class ModelAuthError(ModelError):
    """401/403, or no usable workspace credential. Not transient, never retried."""


class ModelHTTPError(ModelError):
    """Non-200 that retrying will not fix, or that exhausted its attempts."""


class ModelTransportError(ModelError):
    """Connection or timeout failure against the serving endpoint."""


@dataclass(frozen=True)
class ToolCall:
    """One entry of an assistant message's ``tool_calls`` array."""

    id: str
    name: str
    arguments: str
    """Raw JSON text exactly as the model emitted it. Parse it with :meth:`parse_arguments`."""

    def parse_arguments(self) -> dict:
        """Decode :attr:`arguments`, naming the tool when the model emits invalid JSON.

        A model can and does produce malformed tool arguments. The agent loop needs that to be a
        recognisable failure it can report, not a bare ``JSONDecodeError`` from somewhere in the
        stack.
        """
        if not self.arguments.strip():
            return {}
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ModelError(f"tool call {self.name!r} returned arguments that are not JSON: {exc}")
        if not isinstance(parsed, dict):
            raise ModelError(f"tool call {self.name!r} returned {type(parsed).__name__}, not an object")
        return parsed


@dataclass(frozen=True)
class ModelResponse:
    """One chat completion, flattened to what a caller actually uses.

    ``message`` is the raw assistant message dict, kept so the agent loop can append it to the
    conversation verbatim — a re-serialized copy would drop whatever fields the endpoint added.
    """

    task: str
    endpoint: str
    model: str
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    in_tokens: int | None
    out_tokens: int | None
    latency_ms: float
    message: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class Credentials:
    """Where to send the request and how to authenticate it."""

    host: str
    headers: Mapping[str, str]


def resolve_endpoint(task: str, config: Mapping[str, Any] | None = None) -> str:
    """``"agent"`` -> ``model.agent_endpoint``. Unknown task or missing key raises.

    Loudly, in both cases: a typo'd task silently falling back to the agent endpoint would send
    the stretch-goal SLM traffic to a 70B model and nobody would notice until the bill.
    """
    key = TASK_ENDPOINT_KEYS.get(task)
    if key is None:
        raise ValueError(f"unknown task {task!r}, expected one of {sorted(TASK_ENDPOINT_KEYS)}")

    endpoint = config_section("model", config).get(key)
    if not endpoint:
        raise ConfigError(f"config key model.{key} is not set — task {task!r} has no endpoint")
    return str(endpoint)


def workspace_credentials() -> Credentials:
    """Workspace host plus fresh auth headers.

    Environment variables first, so a local run needs nothing but ``DATABRICKS_HOST`` and
    ``DATABRICKS_TOKEN`` from ``.env`` and the tests never import the SDK. Otherwise the SDK's
    ``Config`` resolves the ambient identity — the notebook user, the job's service principal, or
    the Databricks App's — and mints headers per call, which is what keeps an OAuth deployment
    working after the first hour.
    """
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    if host and token:
        return Credentials(host=_normalize_host(host), headers={"Authorization": f"Bearer {token}"})

    try:
        from databricks.sdk.core import Config
    except ImportError as exc:  # pragma: no cover - databricks-sdk is a hard dependency
        raise ModelAuthError(
            "no DATABRICKS_HOST/DATABRICKS_TOKEN in the environment and databricks-sdk is not "
            "installed, so no workspace credential can be resolved"
        ) from exc

    cfg = _sdk_config(Config)
    if not cfg.host:
        raise ModelAuthError(
            "no workspace host: set DATABRICKS_HOST and DATABRICKS_TOKEN locally, or run where "
            "the Databricks SDK can resolve an identity (notebook, job, or app)."
        )
    return Credentials(host=_normalize_host(cfg.host), headers=dict(cfg.authenticate()))


def call_model(
    task: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    response_format: Mapping[str, Any] | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    session: Any | None = None,
    credentials: Credentials | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ModelResponse:
    """Call the Foundation Model endpoint configured for ``task``.

    ``messages``, ``tools`` and ``response_format`` are OpenAI-shaped and are sent unchanged.

    The keyword-only arguments are injection seams, not configuration: ``session`` and
    ``credentials`` let the tests exercise this without HTTP or a workspace, and ``sleep`` /
    ``monotonic`` make "it backed off once" an assertion rather than a two-second test.
    """
    endpoint = resolve_endpoint(task, config)
    payload = _build_payload(messages, tools, response_format)
    creds = credentials or workspace_credentials()
    http = session if session is not None else requests
    url = f"{creds.host}/serving-endpoints/{endpoint}/invocations"

    started_at = monotonic()
    try:
        body = _post_with_retries(http, url, creds, payload, endpoint, timeout, sleep, monotonic)
    except ModelError:
        _record(task, endpoint, (monotonic() - started_at) * 1000.0, ok=False)
        raise

    latency_ms = (monotonic() - started_at) * 1000.0
    response = _parse_response(task, endpoint, body, latency_ms)

    log.info(
        "model call task=%s endpoint=%s status=200 latency_ms=%.0f in_tokens=%s out_tokens=%s "
        "tool_calls=%d finish_reason=%s",
        task,
        endpoint,
        latency_ms,
        response.in_tokens,
        response.out_tokens,
        len(response.tool_calls),
        response.finish_reason,
    )
    _record(
        task,
        response.model,
        latency_ms,
        ok=True,
        in_tokens=response.in_tokens,
        out_tokens=response.out_tokens,
    )
    return response


# ------------------------------------------------------------------------- internals


def _build_payload(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    response_format: Mapping[str, Any] | None,
) -> dict:
    if not messages:
        raise ValueError("messages must not be empty")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or "role" not in message:
            raise ValueError(f"messages[{index}] is not a mapping with a 'role' key")

    payload: dict[str, Any] = {"messages": [dict(message) for message in messages]}
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
    if response_format:
        payload["response_format"] = dict(response_format)
    return payload


def _post_with_retries(
    http: Any,
    url: str,
    creds: Credentials,
    payload: Mapping[str, Any],
    endpoint: str,
    timeout: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Mapping[str, Any]:
    headers = {"Content-Type": "application/json", **creds.headers}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = http.post(url, json=dict(payload), headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            log.warning(
                "model transport failure endpoint=%s attempt=%d/%d error=%s",
                endpoint,
                attempt,
                MAX_ATTEMPTS,
                type(exc).__name__,
            )
            if attempt == MAX_ATTEMPTS:
                raise ModelTransportError(
                    f"{endpoint}: {type(exc).__name__} after {attempt} attempts"
                ) from None
            sleep(_backoff_seconds(attempt))
            continue

        status = response.status_code
        if status == 200:
            body = response.json()
            if not isinstance(body, Mapping):
                raise ModelHTTPError(f"{endpoint}: HTTP 200 with a non-object body")
            return body

        message = _error_message(response)
        log.warning(
            "model call failed endpoint=%s status=%d attempt=%d/%d message=%s",
            endpoint,
            status,
            attempt,
            MAX_ATTEMPTS,
            message,
        )

        if status in (401, 403):
            raise ModelAuthError(
                f"{endpoint}: HTTP {status} — the caller's identity cannot query this serving "
                f"endpoint ({message}). Check CAN QUERY on the endpoint before the code."
            )

        if status in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS:
            sleep(_backoff_seconds(attempt))
            continue

        raise ModelHTTPError(f"{endpoint}: HTTP {status} after {attempt} attempt(s) ({message})")

    raise ModelHTTPError(f"{endpoint}: exhausted {MAX_ATTEMPTS} attempts")


def _parse_response(
    task: str,
    endpoint: str,
    body: Mapping[str, Any],
    latency_ms: float,
) -> ModelResponse:
    choices = body.get("choices") or []
    if not choices:
        raise ModelHTTPError(f"{endpoint}: response contained no choices")

    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    message = message if isinstance(message, Mapping) else {}

    usage = body.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}

    return ModelResponse(
        task=task,
        endpoint=endpoint,
        model=str(body.get("model") or endpoint),
        text=str(message.get("content") or ""),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        finish_reason=choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        in_tokens=_optional_int(usage.get("prompt_tokens")),
        out_tokens=_optional_int(usage.get("completion_tokens")),
        latency_ms=latency_ms,
        message=dict(message),
        raw=dict(body),
    )


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()

    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        function = entry.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=str(entry.get("id") or ""),
                name=str(name),
                arguments=str(function.get("arguments") or ""),
            )
        )
    return tuple(calls)


def _error_message(response: Any) -> str:
    """The endpoint's ``message`` field, or the status class. Never the whole body.

    A serving-endpoint error body is small and does not echo the bearer token the way the Massive
    query-parameter key did (A-1), but it can carry a slice of the prompt back, so only this one
    field is allowed out and it is truncated.
    """
    try:
        payload = response.json()
    except Exception:
        return "no JSON body"
    if not isinstance(payload, Mapping):
        return "non-object body"
    message = payload.get("message") or payload.get("error_code") or payload.get("error")
    if message is None:
        return "no message field"
    text = str(message)
    return text if len(text) <= 200 else text[:197] + "..."


def _record(
    task: str,
    model: str,
    latency_ms: float,
    *,
    ok: bool,
    in_tokens: int | None = None,
    out_tokens: int | None = None,
) -> None:
    """Append one telemetry record. A telemetry failure must never fail the model call."""
    try:
        telemetry_module.record(
            task=task,
            model=model,
            latency_ms=latency_ms,
            ok=ok,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.warning("telemetry record dropped task=%s error=%s", task, type(exc).__name__)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s (+ up to 25% jitter)."""
    base = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    return base + random.uniform(0.0, base * 0.25)


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    return host if "://" in host else f"https://{host}"


_config_singleton: Any | None = None


def _sdk_config(config_class: Any) -> Any:
    """One cached SDK ``Config``; its ``authenticate()`` still mints fresh headers per call.

    Constructing it walks the whole credential-provider chain, which is not something to repeat
    on every turn of a chat.
    """
    global _config_singleton

    if _config_singleton is None:
        _config_singleton = config_class()
    return _config_singleton


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
