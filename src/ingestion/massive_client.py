"""Massive REST API client (spec A-1).

Contract::

    class MassiveClient:
        def __init__(cfg, secret_getter): ...
        def get_daily_aggregates(ticker, start_date, end_date) -> list[dict]
        def get_news(ticker, published_after) -> list[dict]

Requirements:

- Throttle every request from ``cfg.rate_limit_per_min`` (token bucket or sleep-based).
  The limit is configurable, never hard-coded: it depends on the active Massive plan.
- Follow pagination to exhaustion via ``next_url`` from the response envelope. Verified live:
  the envelope is ``{count, next_url, request_id, results, status}``.
- Retry 429 and 5xx with exponential backoff plus jitter, max 5 attempts.
- Raise immediately on 401/403 with a clear message (key or plan problem, not transient).
- Log every request: status, latency, endpoint name, and ``request_id`` when it parses out.

SECURITY — the API key travels as a QUERY PARAMETER, which makes both URLs and error bodies
credential-bearing. On a non-200, NEVER log or print the response body or the full URL. Log the
status code, the ``request_id`` if parseable, and the endpoint name only (e.g.
"reference/news"). Massive's error payloads and any redirect URLs can reflect request params
straight into logs, notebook output, or an agent transcript. The same rule applies in the A-0
smoke test.

Two consequences of that rule are implemented here and are easy to undo by accident:

- ``requests`` exception messages embed the full URL, so transport failures are re-raised as
  ``MassiveTransportError`` carrying only the exception class name, with ``from None`` so the
  original (URL-bearing) message cannot reach a traceback.
- ``redact_secrets`` is exported for callers that persist error text — ``bronze.ingestion_runs``
  stores an ``error`` column, and a stored credential is worse than a logged one.

The API key is never a literal in code. It arrives through the injected ``secret_getter``
callable, backed by Databricks secrets or the environment.

Payload shapes below are VERIFIED live (two independent sessions), not inferred:

- aggregates ``results[]`` = ``{o, h, l, c, v, vw, t, n}`` where ``t`` is epoch-milliseconds at
  the start of the exchange session (04:00Z == 00:00 America/New_York).
- news ``results[]`` = ``{id, title, description, publisher{name, homepage_url, logo_url,
  favicon_url}, article_url, author, published_utc, tickers[], insights[], keywords[]}`` with
  ``insights[]`` = ``{ticker, sentiment, sentiment_reasoning}`` and NO numeric score.

Routes verified live: ``/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`` and
``/v2/reference/news``, both against base ``https://api.massive.com``.

This module returns near-raw vendor dicts. The only annotation it adds is ``_request_id``
(:data:`REQUEST_ID_KEY`), stamped onto every result from its page envelope, because bronze must
store ``request_id`` per row (A-2) and the envelope is otherwise discarded by the
``list[dict]`` return type.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any

import requests

log = logging.getLogger(__name__)

#: Value written to the bronze ``source`` column.
SOURCE = "massive"

#: Key under which the page envelope's ``request_id`` is stamped onto each result dict.
REQUEST_ID_KEY = "_request_id"

DEFAULT_BASE_URL = "https://api.massive.com"

# Endpoint LABELS, used for logging and error messages. Never log a URL: the key is a query
# parameter, so the URL is a credential.
AGGREGATES_ENDPOINT = "aggs/ticker"
NEWS_ENDPOINT = "reference/news"

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_TIMEOUT_SECONDS = 30

# Pagination guard: next_url is vendor-controlled, so a cycle or a runaway cursor must not turn
# into an unbounded loop against a rate-limited API.
MAX_PAGES = 500

_APIKEY_PARAM_RE = re.compile(r"(apikey=)[^&\s\"']+", re.IGNORECASE)


class MassiveError(RuntimeError):
    """Base class for every Massive client failure."""


class MassiveAuthError(MassiveError):
    """401/403 — the key or the plan is wrong. Not transient, never retried."""


class MassiveHTTPError(MassiveError):
    """Non-200 that retrying will not fix, or that exhausted its attempts."""


class MassiveTransportError(MassiveError):
    """Connection/timeout failure. Carries no URL — see the module security note."""


def redact_secrets(text: str) -> str:
    """Mask any ``apiKey=`` value in ``text``.

    For error strings that are logged or persisted. Callers should still prefer never to build
    such a string from a response body or URL in the first place; this is the backstop for
    exception text raised by libraries that do not know the key is sensitive.
    """
    return _APIKEY_PARAM_RE.sub(r"\1***REDACTED***", text)


def env_secret_getter(var_name: str = "MASSIVE_API_KEY") -> Callable[[], str]:
    """Secret getter backed by an environment variable (local development).

    Databricks jobs and notebooks should pass the secret-scope getter instead::

        secret_getter=lambda: dbutils.secrets.get(scope="capstone", key="massive_api_key")
    """

    def _get() -> str:
        key = os.environ.get(var_name)
        if not key:
            raise MassiveAuthError(
                f"No Massive API key: environment variable {var_name} is unset. Set it locally "
                "(.env), or pass secret_getter=lambda: dbutils.secrets.get(scope='capstone', "
                "key='massive_api_key') when running on Databricks."
            )
        return key

    return _get


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute-style config object.

    The spec writes ``cfg.rate_limit_per_min`` while ``config/config.yaml`` loads as nested
    dicts, so both access styles are accepted.
    """
    if isinstance(cfg, Mapping):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _as_iso_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class MassiveClient:
    """Throttled, paginating, retrying client for the Massive market-data API.

    ``cfg`` is the ``massive`` section of ``config/config.yaml``. ``secret_getter`` is a
    zero-argument callable returning the API key; it is called once and cached.

    ``session``, ``sleep`` and ``monotonic`` are injectable so the throttle, the backoff and the
    pagination walk are testable without real HTTP or real waiting.
    """

    def __init__(
        self,
        cfg: Any,
        secret_getter: Callable[[], str],
        *,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base_url = str(_cfg_get(cfg, "base_url", DEFAULT_BASE_URL)).rstrip("/")
        rate_limit_per_min = float(_cfg_get(cfg, "rate_limit_per_min", 5))
        if rate_limit_per_min <= 0:
            raise ValueError("massive.rate_limit_per_min must be greater than 0")
        self._min_interval_seconds = 60.0 / rate_limit_per_min
        self._secret_getter = secret_getter
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._api_key: str | None = None

    # ------------------------------------------------------------------ public API

    def get_daily_aggregates(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str,
    ) -> list[dict]:
        """Daily OHLCV bars for ``ticker`` over an inclusive date range.

        Each returned dict is a near-raw ``results[]`` entry (``{o, h, l, c, v, vw, t, n}``)
        stamped with :data:`REQUEST_ID_KEY`.
        """
        url = (
            f"{self._base_url}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{_as_iso_date(start_date)}/{_as_iso_date(end_date)}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        return self._get_all_pages(AGGREGATES_ENDPOINT, url, params)

    def get_news(self, ticker: str, published_after: date | datetime | str) -> list[dict]:
        """News articles for ``ticker`` published at or after ``published_after``.

        Each returned dict is a near-raw article (nested ``publisher``, ``tickers[]``,
        ``insights[]``) stamped with :data:`REQUEST_ID_KEY`. No filtering, flattening or
        sentiment mapping happens here — that is silver's job (A-3).
        """
        params = {
            # TODO: the ROUTE and the payload shape are verified live, but this filter parameter
            # name is not. Confirm against Massive's docs. If it is ignored the run over-fetches
            # rather than under-fetching, and the (article_id, ticker) MERGE absorbs the overlap,
            # so a wrong name here costs requests, not correctness.
            "published_utc.gte": _as_published_after(published_after),
            "ticker": ticker,
            "order": "asc",
            "sort": "published_utc",
            "limit": 1000,
        }
        return self._get_all_pages(NEWS_ENDPOINT, f"{self._base_url}/v2/reference/news", params)

    # ------------------------------------------------------------------ internals

    def _key(self) -> str:
        if self._api_key is None:
            self._api_key = self._secret_getter()
        return self._api_key

    def _get_all_pages(self, endpoint: str, url: str, params: dict) -> list[dict]:
        """Walk ``next_url`` to exhaustion, returning every result in page order."""
        results: list[dict] = []
        next_url: str | None = url
        page_params: dict | None = params
        seen: set[str] = set()
        pages = 0

        while next_url:
            payload = self._request(endpoint, next_url, page_params or {})
            request_id = payload.get("request_id")
            for result in payload.get("results") or []:
                if isinstance(result, dict):
                    result[REQUEST_ID_KEY] = request_id
                    results.append(result)

            pages += 1
            seen.add(next_url)
            next_url = payload.get("next_url") or None
            # next_url comes back WITHOUT the key (it is a query parameter, and the vendor does
            # not echo it), so the cursor page must re-attach it; _request does that.
            page_params = None

            if next_url and next_url in seen:
                log.warning(
                    "massive pagination stopped: next_url repeated endpoint=%s pages=%d",
                    endpoint,
                    pages,
                )
                break
            if pages >= MAX_PAGES:
                log.warning(
                    "massive pagination stopped at MAX_PAGES endpoint=%s pages=%d", endpoint, pages
                )
                break

        log.info(
            "massive fetch complete endpoint=%s pages=%d results=%d", endpoint, pages, len(results)
        )
        return results

    def _request(self, endpoint: str, url: str, params: dict) -> dict:
        """One throttled request with retries. Returns the parsed 200 envelope."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            started_at = self._monotonic()
            try:
                response = self._session.get(
                    url,
                    params={**params, "apiKey": self._key()},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                # The exception message embeds the full URL, which carries the key. Log the class
                # name only, and re-raise `from None` so no traceback can print the original.
                log.warning(
                    "massive transport failure endpoint=%s attempt=%d/%d error=%s",
                    endpoint,
                    attempt,
                    MAX_ATTEMPTS,
                    type(exc).__name__,
                )
                if attempt == MAX_ATTEMPTS:
                    raise MassiveTransportError(
                        f"{endpoint}: {type(exc).__name__} after {attempt} attempts"
                    ) from None
                self._sleep(self._backoff_seconds(attempt))
                continue

            latency_ms = (self._monotonic() - started_at) * 1000.0
            status = response.status_code

            if status == 200:
                payload = response.json()
                log.info(
                    "massive request endpoint=%s status=200 latency_ms=%.0f request_id=%s",
                    endpoint,
                    latency_ms,
                    payload.get("request_id") if isinstance(payload, Mapping) else None,
                )
                return payload if isinstance(payload, dict) else {}

            # Non-200: status, request_id and endpoint label ONLY. No body, no URL.
            request_id = _request_id_of(response)
            log.warning(
                "massive request endpoint=%s status=%d latency_ms=%.0f request_id=%s",
                endpoint,
                status,
                latency_ms,
                request_id,
            )

            if status in (401, 403):
                raise MassiveAuthError(
                    f"{endpoint}: HTTP {status} — Massive rejected the API key or the plan does "
                    f"not cover this endpoint (request_id={request_id}). Response body withheld: "
                    "it can reflect the key back."
                )

            if status in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS:
                self._sleep(self._backoff_seconds(attempt))
                continue

            raise MassiveHTTPError(
                f"{endpoint}: HTTP {status} after {attempt} attempt(s) "
                f"(request_id={request_id})"
            )

        raise MassiveHTTPError(f"{endpoint}: exhausted {MAX_ATTEMPTS} attempts")

    def _throttle(self) -> None:
        """Sleep so consecutive requests are at least ``60 / rate_limit_per_min`` apart.

        If backfill starts returning 429s, this is the first thing to check (spec C-e): log the
        inter-request gaps and confirm the sleep is actually happening.
        """
        if self._last_request_at is not None:
            elapsed = self._monotonic() - self._last_request_at
            wait = self._min_interval_seconds - elapsed
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff with jitter: 1s, 2s, 4s, 8s (+ up to 25% jitter)."""
        base = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        return base + random.uniform(0.0, base * 0.25)


def _as_published_after(value: date | datetime | str) -> str:
    """Render a news watermark as the ISO-8601 UTC string the API filters on."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return f"{value.isoformat()}T00:00:00Z"
    return str(value)


def _request_id_of(response: Any) -> str | None:
    """Pull ``request_id`` out of an error payload without exposing the payload.

    Reading the body is fine; logging it is not. Only this one field escapes.
    """
    try:
        payload = response.json()
    except Exception:
        return None
    if isinstance(payload, Mapping):
        request_id = payload.get("request_id")
        return str(request_id) if request_id is not None else None
    return None
