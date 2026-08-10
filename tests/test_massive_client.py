"""MassiveClient tests (spec A-1). Mocked HTTP only — no network, no real sleeping.

Covers the five A-1 behaviours plus the security rule:

- pagination follows ``next_url`` to exhaustion and re-attaches the key on cursor pages
- 429 retries with exponential backoff, then succeeds
- 5xx exhausts ``MAX_ATTEMPTS`` and raises
- 401/403 raise immediately, with NOTHING credential-bearing in the log or the message
- the throttle actually spaces requests by ``60 / rate_limit_per_min``

The clock and ``sleep`` are injected, so "backoff waited ~1s" is asserted rather than endured.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pytest
import requests

from src.ingestion.massive_client import (
    BACKOFF_BASE_SECONDS,
    MAX_ATTEMPTS,
    NEWS_ENDPOINT,
    REQUEST_ID_KEY,
    MassiveAuthError,
    MassiveClient,
    MassiveHTTPError,
    MassiveTransportError,
    env_secret_getter,
    redact_secrets,
)
from tests.conftest import (
    SIMPLE_ARTICLE_ID,
    STRICT_SUBSET_ARTICLE_ID,
    UNKNOWN_LABEL_ARTICLE_ID,
)

API_KEY = "sk-live-NEVER-LOG-ME-1234567890"
BASE_URL = "https://api.massive.com"
CFG = {"base_url": BASE_URL, "rate_limit_per_min": 5}

# A real 401 body, and a body that reflects the key back the way an error page or redirect can.
# Neither may ever reach a log record or an exception message.
REAL_401_BODY = '{"status":"ERROR","request_id":"af5d2de41ff66d45d7449f1849c711ab","error":"API Key was not provided"}'
REFLECTING_BODY = f'{{"error":"bad request for /v2/reference/news?ticker=NVDA&apiKey={API_KEY}"}}'


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
    text: str = ""
    json_raises: bool = False

    def json(self) -> object:
        if self.json_raises:
            raise ValueError("not json")
        return self.payload


@dataclass
class RecordedCall:
    url: str
    params: dict
    at: float


@dataclass
class FakeSession:
    """Returns queued responses (or raises queued exceptions) and records every call."""

    responses: list
    clock: FakeClock | None = None
    calls: list[RecordedCall] = field(default_factory=list)

    def get(self, url, params=None, timeout=None):
        self.calls.append(
            RecordedCall(url=url, params=dict(params or {}), at=self.clock.now if self.clock else 0.0)
        )
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_client(responses, *, cfg=None, clock=None, secret_getter=None):
    clock = clock or FakeClock()
    session = FakeSession(responses=list(responses), clock=clock)
    client = MassiveClient(
        cfg if cfg is not None else CFG,
        secret_getter or (lambda: API_KEY),
        session=session,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return client, session, clock


def page(results, next_url=None, request_id="rid-1"):
    return FakeResponse(
        status_code=200,
        payload={
            "count": len(results),
            "status": "OK",
            "request_id": request_id,
            "next_url": next_url,
            "results": results,
        },
    )


# --------------------------------------------------------------------- pagination


def test_pagination_follows_next_url_to_exhaustion():
    responses = [
        page([{"c": 1.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P2", request_id="rid-p1"),
        page([{"c": 2.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P3", request_id="rid-p2"),
        page([{"c": 3.0}], next_url=None, request_id="rid-p3"),
    ]
    client, session, _ = make_client(responses)

    results = client.get_news("NVDA", date(2024, 8, 1))

    assert [r["c"] for r in results] == [1.0, 2.0, 3.0]
    assert len(session.calls) == 3
    # Every result carries the request_id of the page it came from — bronze stores it per row.
    assert [r[REQUEST_ID_KEY] for r in results] == ["rid-p1", "rid-p2", "rid-p3"]


def test_cursor_pages_reattach_the_key_and_drop_first_page_params():
    responses = [
        page([{"c": 1.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P2"),
        page([{"c": 2.0}]),
    ]
    client, session, _ = make_client(responses)

    client.get_news("NVDA", date(2024, 8, 1))

    first, second = session.calls
    assert first.params["ticker"] == "NVDA"
    assert first.params["apiKey"] == API_KEY
    # next_url already encodes the query; the vendor does not echo the key, so the cursor page
    # carries the key and nothing else.
    assert second.url.endswith("cursor=P2")
    assert second.params == {"apiKey": API_KEY}


def test_pagination_stops_when_next_url_repeats():
    cycle = f"{BASE_URL}/v2/reference/news?cursor=LOOP"
    responses = [page([{"c": 1.0}], next_url=cycle), page([{"c": 2.0}], next_url=cycle)]
    client, session, _ = make_client(responses)

    results = client.get_news("NVDA", date(2024, 8, 1))

    assert len(results) == 2
    assert len(session.calls) == 2  # would loop forever without the cycle guard


def test_secret_getter_is_called_once_across_pages():
    calls = []

    def getter():
        calls.append(1)
        return API_KEY

    responses = [
        page([{"c": 1.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P2"),
        page([{"c": 2.0}]),
    ]
    client, _, _ = make_client(responses, secret_getter=getter)

    client.get_news("NVDA", date(2024, 8, 1))

    assert len(calls) == 1


# ------------------------------------------------------------------ retry/backoff


def test_429_then_success_backs_off_once():
    # A high rate limit keeps throttle sleeps out of the way so the recorded sleep is the backoff.
    cfg = {"base_url": BASE_URL, "rate_limit_per_min": 6000}
    responses = [FakeResponse(429, payload={"request_id": "rid-429"}), page([{"c": 1.0}])]
    client, session, clock = make_client(responses, cfg=cfg)

    results = client.get_news("NVDA", date(2024, 8, 1))

    assert len(results) == 1
    assert len(session.calls) == 2
    assert len(clock.sleeps) == 1
    assert BACKOFF_BASE_SECONDS <= clock.sleeps[0] <= BACKOFF_BASE_SECONDS * 1.25


def test_5xx_exhausts_max_attempts_then_raises():
    cfg = {"base_url": BASE_URL, "rate_limit_per_min": 6000}
    responses = [FakeResponse(503, payload={"request_id": "rid-503"}) for _ in range(MAX_ATTEMPTS)]
    client, session, clock = make_client(responses, cfg=cfg)

    with pytest.raises(MassiveHTTPError) as excinfo:
        client.get_news("NVDA", date(2024, 8, 1))

    assert len(session.calls) == MAX_ATTEMPTS
    assert len(clock.sleeps) == MAX_ATTEMPTS - 1
    # Exponential, not constant: each wait is at least double the previous base.
    assert clock.sleeps == sorted(clock.sleeps)
    assert clock.sleeps[-1] >= BACKOFF_BASE_SECONDS * 2 ** (MAX_ATTEMPTS - 2)
    assert NEWS_ENDPOINT in str(excinfo.value)
    assert "rid-503" in str(excinfo.value)


def test_transport_failure_retries_then_raises_without_url():
    cfg = {"base_url": BASE_URL, "rate_limit_per_min": 6000}
    leaky = requests.ConnectionError(
        f"HTTPSConnectionPool(host='api.massive.com', port=443): url: "
        f"/v2/reference/news?ticker=NVDA&apiKey={API_KEY}"
    )
    client, session, _ = make_client([leaky for _ in range(MAX_ATTEMPTS)], cfg=cfg)

    with pytest.raises(MassiveTransportError) as excinfo:
        client.get_news("NVDA", date(2024, 8, 1))

    message = str(excinfo.value)
    assert len(session.calls) == MAX_ATTEMPTS
    assert API_KEY not in message
    assert "api.massive.com" not in message
    # `from None` keeps the URL-bearing original out of any traceback.
    assert excinfo.value.__cause__ is None


# --------------------------------------------------------------------- auth + leaks


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_raise_immediately(status):
    responses = [
        FakeResponse(
            status,
            payload={"status": "ERROR", "request_id": "rid-auth", "error": "API Key was not provided"},
            text=REAL_401_BODY,
        )
    ]
    client, session, clock = make_client(responses)

    with pytest.raises(MassiveAuthError) as excinfo:
        client.get_news("NVDA", date(2024, 8, 1))

    assert len(session.calls) == 1  # never retried
    assert clock.sleeps == []
    assert str(status) in str(excinfo.value)
    assert "rid-auth" in str(excinfo.value)


def test_non_200_logs_status_request_id_endpoint_and_nothing_else(caplog):
    responses = [FakeResponse(401, payload={"request_id": "rid-auth"}, text=REFLECTING_BODY)]
    client, _, _ = make_client(responses)

    with caplog.at_level(logging.DEBUG, logger="src.ingestion.massive_client"):
        with pytest.raises(MassiveAuthError) as excinfo:
            client.get_news("NVDA", date(2024, 8, 1))

    logged = caplog.text
    assert "status=401" in logged
    assert "rid-auth" in logged
    assert NEWS_ENDPOINT in logged

    # The A-1 security rule, asserted rather than trusted: no key, no URL, no body.
    for forbidden in (API_KEY, "apiKey=", "api.massive.com", REFLECTING_BODY):
        assert forbidden not in logged, f"leaked into logs: {forbidden!r}"
        assert forbidden not in str(excinfo.value), f"leaked into the exception: {forbidden!r}"


def test_error_payload_without_json_still_logs_and_raises(caplog):
    responses = [FakeResponse(500, json_raises=True, text="<html>gateway error</html>") for _ in range(MAX_ATTEMPTS)]
    cfg = {"base_url": BASE_URL, "rate_limit_per_min": 6000}
    client, _, _ = make_client(responses, cfg=cfg)

    with caplog.at_level(logging.DEBUG, logger="src.ingestion.massive_client"):
        with pytest.raises(MassiveHTTPError):
            client.get_news("NVDA", date(2024, 8, 1))

    assert "status=500" in caplog.text
    assert "gateway error" not in caplog.text


def test_redact_secrets_masks_the_query_parameter():
    dirty = f"GET /v2/reference/news?ticker=NVDA&apiKey={API_KEY}&limit=1000 failed"
    clean = redact_secrets(dirty)

    assert API_KEY not in clean
    assert "apiKey=***REDACTED***" in clean
    assert "ticker=NVDA" in clean  # only the credential is masked


# ------------------------------------------------------------------------ throttle


def test_throttle_spaces_requests_by_the_configured_rate_limit():
    cfg = {"base_url": BASE_URL, "rate_limit_per_min": 5}  # 12s between requests
    responses = [
        page([{"c": 1.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P2"),
        page([{"c": 2.0}], next_url=f"{BASE_URL}/v2/reference/news?cursor=P3"),
        page([{"c": 3.0}]),
    ]
    client, session, _ = make_client(responses, cfg=cfg)

    client.get_news("NVDA", date(2024, 8, 1))

    gaps = [b.at - a.at for a, b in zip(session.calls, session.calls[1:])]
    assert gaps == pytest.approx([12.0, 12.0])


def test_rate_limit_must_be_positive():
    with pytest.raises(ValueError):
        MassiveClient({"base_url": BASE_URL, "rate_limit_per_min": 0}, lambda: API_KEY)


def test_attribute_style_config_is_accepted():
    class Cfg:
        base_url = BASE_URL
        rate_limit_per_min = 30

    clock = FakeClock()
    client = MassiveClient(
        Cfg(), lambda: API_KEY, session=FakeSession([page([])], clock), sleep=clock.sleep, monotonic=clock.monotonic
    )

    assert client.get_news("NVDA", date(2024, 8, 1)) == []


# ------------------------------------------------------------- request construction


def test_get_daily_aggregates_builds_the_verified_route(aggregates_envelope):
    client, session, _ = make_client([FakeResponse(200, payload=aggregates_envelope)])

    results = client.get_daily_aggregates("NVDA", date(2026, 7, 1), date(2026, 8, 1))

    call = session.calls[0]
    assert call.url == f"{BASE_URL}/v2/aggs/ticker/NVDA/range/1/day/2026-07-01/2026-08-01"
    assert call.params["adjusted"] == "true"
    assert call.params["sort"] == "asc"
    # Near-raw pass-through: the vendor's single-letter keys survive untouched.
    assert set(results[0]) >= {"o", "h", "l", "c", "v", "vw", "t", "n"}
    assert results[0][REQUEST_ID_KEY] == aggregates_envelope["request_id"]


@pytest.mark.parametrize(
    "published_after,expected",
    [
        (date(2024, 8, 1), "2024-08-01T00:00:00Z"),
        (datetime(2026, 8, 9, 13, 45, 30, tzinfo=timezone.utc), "2026-08-09T13:45:30Z"),
        ("2026-08-09T13:45:30Z", "2026-08-09T13:45:30Z"),
    ],
)
def test_get_news_sends_the_watermark_as_iso_utc(published_after, expected, news_envelope):
    client, session, _ = make_client([FakeResponse(200, payload=news_envelope)])

    client.get_news("NVDA", published_after)

    call = session.calls[0]
    assert call.url == f"{BASE_URL}/v2/reference/news"
    assert call.params["published_utc.gte"] == expected
    assert call.params["ticker"] == "NVDA"


# ------------------------------------------------- vendor contract pinned by fixtures


def test_fixtures_match_the_verified_news_shape(news_results):
    assert len(news_results) == 3
    for article in news_results:
        assert re.fullmatch(r"[0-9a-f]{64}", article["id"]), "article id must be a 64-hex digest"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", article["published_utc"])
        assert isinstance(article["publisher"], dict)
        assert article["publisher"]["name"]  # publisher is nested; the mapping takes .name
        for insight in article["insights"]:
            # Exactly these three keys. Massive returns NO numeric score; sentiment_score is
            # derived at silver build time (A-3).
            assert set(insight) == {"ticker", "sentiment", "sentiment_reasoning"}
            assert not any(isinstance(v, (int, float)) for v in insight.values())


def test_fixtures_include_a_strict_subset_insights_article(news_results):
    article = next(a for a in news_results if a["id"] == STRICT_SUBSET_ARTICLE_ID)
    insight_tickers = {i["ticker"] for i in article["insights"]}

    assert insight_tickers < set(article["tickers"])
    assert {"TSLA", "WDC"} == set(article["tickers"]) - insight_tickers


def test_fixtures_include_an_unrecognized_sentiment_label(news_results):
    article = next(a for a in news_results if a["id"] == UNKNOWN_LABEL_ARTICLE_ID)

    assert article["insights"][0]["sentiment"] not in {"positive", "neutral", "negative"}


def test_aggregates_fixture_volume_is_a_float(aggregates_results):
    # v arrives in scientific notation, which is why bronze stores volume as DOUBLE, not BIGINT.
    assert all(isinstance(bar["v"], float) for bar in aggregates_results)
    assert all(isinstance(bar["t"], int) for bar in aggregates_results)


def test_simple_fixture_article_is_single_insight(news_results):
    article = next(a for a in news_results if a["id"] == SIMPLE_ARTICLE_ID)

    assert [i["ticker"] for i in article["insights"]] == article["tickers"]


# ---------------------------------------------------------------------- secret getter


def test_env_secret_getter_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", API_KEY)

    assert env_secret_getter()() == API_KEY


def test_env_secret_getter_raises_a_clear_error_when_unset(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    with pytest.raises(MassiveAuthError) as excinfo:
        env_secret_getter()()

    message = str(excinfo.value)
    assert "MASSIVE_API_KEY" in message
    assert "dbutils.secrets" in message  # points at the Databricks path, not just "failed"
