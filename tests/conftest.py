"""Shared fixtures: REAL Massive payload shapes (A-1/A-2/A-3/A-5) and the Checkpoint B series.

PART TWO, added at B-1/B-2, is at the bottom of this file: the synthetic two-regime return
series the modeling tests fit, and the fitted results themselves. The fits are session-scoped
because a ``search_reps=20`` MLE on 800 observations costs seconds and ``tests/test_models.py``,
``tests/test_monte_carlo.py`` and ``tests/test_no_lookahead.py`` all need the same one. Nothing
in a test may mutate them.

PART ONE (spec A-1, A-2, A-3, A-5).

Fixtures are the only place the vendor contract is pinned in code, so a fixture that disagrees
with production is worse than no fixture: it makes wrong code pass. Everything here was taken
from live 200 responses (two independent sessions), not from memory:

- aggregates ``results[]`` = ``{v, vw, o, c, h, l, t, n}``; ``v`` arrives as a FLOAT in
  scientific notation, and ``t`` is epoch-milliseconds at the start of the exchange session
  (04:00Z == 00:00 America/New_York, so the UTC and session dates agree for daily bars).
- news articles carry a NESTED ``publisher`` dict, a 64-char hex ``id``, an ISO-8601 UTC
  ``published_utc`` string, and ``insights[]`` entries with exactly
  ``{ticker, sentiment, sentiment_reasoning}`` — there is NO numeric sentiment score in the
  payload. ``sentiment_score`` is derived at silver build time (A-3).

``news_envelope`` deliberately contains the two articles the A-5 fixture rules require:

- ``STRICT_SUBSET_ARTICLE_ID`` lists four tickers but only two insights, so ``TSLA`` and ``WDC``
  must produce NO row. That is the A-3 explode rule, and it is the one that silently regresses
  if someone "simplifies" the explode back to the ``tickers`` array.
- ``UNKNOWN_LABEL_ARTICLE_ID`` carries an unrecognized sentiment label, for the A-3 assertion
  that it degrades to ``sentiment_score`` 0 plus a logged warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Put the repo root on sys.path so tests import `src.*` exactly as the notebooks do (spec C-a),
# without needing an installed package or a pytest ini file.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: Article listing 4 tickers with only 2 insights (insights ⊂ tickers). Verified live shape.
STRICT_SUBSET_ARTICLE_ID = "e3459839b803ce2dc723665e6465c2efeb2f6d3b9793d6386bfbb62ce2dea383"

#: Article whose sentiment label is outside {positive, neutral, negative}.
UNKNOWN_LABEL_ARTICLE_ID = "b41c7d5e9f2a08c36d1e4b7a09f8c2d5e6a3b0c9d8e7f61524a3b2c1d0e9f8a7"

#: Ordinary single-insight article.
SIMPLE_ARTICLE_ID = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809"


@pytest.fixture
def aggregates_results() -> list[dict]:
    """Three consecutive NVDA daily bars, copied from a live 200 response."""
    return [
        {
            "v": 1.46147597081851e08,
            "vw": 197.0727,
            "o": 196.2,
            "c": 197.58,
            "h": 199.85,
            "l": 193.45,
            "t": 1782878400000,  # 2026-07-01 00:00 America/New_York
            "n": 2330312,
        },
        {
            "v": 1.42385548018339e08,
            "vw": 195.0961,
            "o": 197.14,
            "c": 194.83,
            "h": 200.055,
            "l": 192.35,
            "t": 1782964800000,  # 2026-07-02
            "n": 2549201,
        },
        {
            "v": 1.08999015263052e08,
            "vw": 195.9732,
            "o": 194.42,
            "c": 195.55,
            "h": 197.55,
            "l": 193.99,
            "t": 1783310400000,  # 2026-07-06 (weekend gap: no bars for the 4th/5th)
            "n": 2212755,
        },
    ]


@pytest.fixture
def aggregates_envelope(aggregates_results: list[dict]) -> dict:
    """Aggregates page envelope: ``{count, next_url, request_id, results, status}``."""
    return {
        "ticker": "NVDA",
        "queryCount": 3,
        "resultsCount": 3,
        "adjusted": True,
        "count": 3,
        "status": "OK",
        "request_id": "6a7fbd2f4f0c4b0e8a1d5c9e3b7f2a10",
        "results": aggregates_results,
    }


@pytest.fixture
def news_results() -> list[dict]:
    """Three news articles, shaped exactly as the live endpoint returns them."""
    return [
        {
            "id": STRICT_SUBSET_ARTICLE_ID,
            "publisher": {
                "name": "The Motley Fool",
                "homepage_url": "https://www.fool.com/",
                "logo_url": "https://s3.massive.com/public/assets/news/logos/themotleyfool.svg",
                "favicon_url": "https://s3.massive.com/public/assets/news/favicons/themotleyfool.ico",
            },
            "title": "Sandisk Has Surged More Than 3,000% in 12 Months. Is a Stock Split Coming?",
            "author": "Micah Zimmerman",
            "published_utc": "2026-08-10T02:15:00Z",
            "article_url": "https://www.fool.com/investing/2026/08/09/sandisk-has-surged-more-than-3000-in-12-months-is/",
            "tickers": ["SNDK", "NVDA", "TSLA", "WDC"],
            "image_url": "https://g.foolcdn.com/image/?url=example.jpg&w=1200&op=resize",
            "description": "Shares of the memory maker have soared alongside AI datacenter demand.",
            "keywords": ["investing", "semiconductors"],
            # insights ⊂ tickers: TSLA and WDC appear in tickers but have NO insight, so the
            # A-3 explode must yield no row for them.
            "insights": [
                {
                    "ticker": "SNDK",
                    "sentiment": "positive",
                    "sentiment_reasoning": "The article highlights a 3,000% surge and speculates about a split.",
                },
                {
                    "ticker": "NVDA",
                    "sentiment": "neutral",
                    "sentiment_reasoning": "Nvidia is mentioned only as context for datacenter demand.",
                },
            ],
        },
        {
            "id": UNKNOWN_LABEL_ARTICLE_ID,
            "publisher": {
                "name": "Benzinga",
                "homepage_url": "https://www.benzinga.com/",
                "logo_url": "https://s3.massive.com/public/assets/news/logos/benzinga.svg",
                "favicon_url": "https://s3.massive.com/public/assets/news/favicons/benzinga.ico",
            },
            "title": "Nvidia Slips After Guidance, Analysts Split On What Comes Next",
            "author": "Staff Writer",
            "published_utc": "2026-08-09T13:45:30Z",
            "article_url": "https://www.benzinga.com/news/2026/08/nvidia-guidance",
            "tickers": ["NVDA"],
            "image_url": "https://cdn.benzinga.com/files/example.jpeg",
            "description": "Analysts disagree on whether the guidance implies a demand plateau.",
            "keywords": ["earnings"],
            # Unrecognized label: must degrade to sentiment_score 0 with a warning at silver
            # build time, never fail the run and never be silently dropped.
            "insights": [
                {
                    "ticker": "NVDA",
                    "sentiment": "mixed",
                    "sentiment_reasoning": "The article presents both bullish and bearish analyst takes.",
                }
            ],
        },
        {
            "id": SIMPLE_ARTICLE_ID,
            "publisher": {
                "name": "Reuters",
                "homepage_url": "https://www.reuters.com/",
                "logo_url": "https://s3.massive.com/public/assets/news/logos/reuters.svg",
                "favicon_url": "https://s3.massive.com/public/assets/news/favicons/reuters.ico",
            },
            "title": "Microsoft Expands Datacenter Footprint",
            "author": "Reuters Staff",
            "published_utc": "2026-08-08T20:05:00Z",
            "article_url": "https://www.reuters.com/technology/microsoft-datacenter",
            "tickers": ["MSFT"],
            "image_url": "https://cloudfront.reuters.com/example.jpg",
            "description": "The company announced three new regions.",
            "keywords": ["cloud"],
            "insights": [
                {
                    "ticker": "MSFT",
                    "sentiment": "positive",
                    "sentiment_reasoning": "Expansion is framed as evidence of durable cloud demand.",
                }
            ],
        },
    ]


@pytest.fixture
def news_envelope(news_results: list[dict]) -> dict:
    """News page envelope: same ``{count, next_url, request_id, results, status}`` shape."""
    return {
        "count": 3,
        "status": "OK",
        "request_id": "af5d2de41ff66d45d7449f1849c711ab",
        "next_url": None,
        "results": news_results,
    }


# ------------------------------------------------------- Checkpoint B synthetic series
# A two-regime series with KNOWN parameters, so a fit can be checked against the truth rather
# than against itself. Everything is in the percent scale the models estimate in (spec B-0):
# N(0, 1%) and N(0, 3%) daily returns are sigma 1.0 and 3.0 here.

#: True regime standard deviations, percent scale. Well separated on purpose: the point of the
#: recovery test is the low/high ORDERING after sorting, not how small a difference MLE can find.
TRUE_SIGMAS_PCT = (1.0, 3.0)

#: True regime means, percent scale. Calm drifts up slightly, turbulent drifts down — the usual
#: empirical asymmetry, and it makes a mislabelled regime visible in the means too.
TRUE_MUS_PCT = (0.05, -0.15)

#: True probability of staying in the current regime. Persistent enough to be identifiable, far
#: enough from 1 to stay clear of the B-2 absorbing-state check.
TRUE_STAY_PROBABILITY = 0.96

#: Roughly three years of trading days: comfortably above ``backtest.min_train_days`` (252).
SERIES_LENGTH = 800

#: Fixed so every assertion below is reproducible. A modeling test that passes on some seeds is
#: not a test.
SERIES_SEED = 20260810


def two_regime_series(
    length: int = SERIES_LENGTH, seed: int = SERIES_SEED
) -> tuple["object", "object"]:
    """Percent log returns from a known two-regime Markov chain, plus the true regime path.

    The chain is simulated with the LEFT-stochastic convention the whole repo uses, written out
    longhand rather than through the production sampler: a fixture that generates its data with
    the code under test cannot catch a transposed sampler.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    states = np.zeros(length, dtype=int)
    for t in range(1, length):
        stays = rng.random() < TRUE_STAY_PROBABILITY
        states[t] = states[t - 1] if stays else 1 - states[t - 1]

    mus = np.asarray(TRUE_MUS_PCT, dtype=float)[states]
    sigmas = np.asarray(TRUE_SIGMAS_PCT, dtype=float)[states]
    return rng.normal(mus, sigmas), states


def regime_news_series(states, seed: int = SERIES_SEED + 1) -> "object":
    """A ``news_sentiment_3d``-shaped series that carries real information about the regime.

    Bounded in [-1, 1] like the real column (A-4: a decayed mean of +1/0/-1 scores), and negative
    in the turbulent regime, so Model C's transition model has something to find. Model C is not
    presumed to win, but a TVTP fixture built on pure noise would test the plumbing only.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    signal = -0.6 * np.asarray(states, dtype=float)
    return np.clip(signal + rng.normal(0.0, 0.3, size=signal.size), -1.0, 1.0)


def features_frame(returns_pct, news, *, start=None, start_price: float = 100.0):
    """A ``silver.daily_features``-shaped pandas frame for one ticker (spec B-5 input).

    Shaped like what ``.toPandas()`` hands the modeling layer, including the parts that trip code
    up: a LEADING NULL ``log_return`` on the first row, because there is no previous close (A-4),
    and one row per business day so ISO weeks have five sessions. ``news_sentiment_3d`` is aligned
    one-for-one with ``returns_pct`` from row 1, matching how the returns and the news column are
    read from the same rows before Model C applies its lag.

    The extra ``ticker`` column is deliberate: the real frame carries columns the backtest does not
    read, and it must tolerate them.
    """
    import numpy as np
    import pandas as pd

    returns = np.asarray(returns_pct, dtype=float)
    log_return = np.concatenate([[np.nan], returns / 100.0])
    close = start_price * np.exp(np.concatenate([[0.0], np.cumsum(returns / 100.0)]))
    return pd.DataFrame(
        {
            "ticker": "NVDA",
            "trade_date": pd.bdate_range(start or "2023-01-02", periods=log_return.size).date,
            "close": close,
            "log_return": log_return,
            "news_sentiment_3d": np.concatenate([[0.0], np.asarray(news, dtype=float)]),
        }
    )


def backtest_config(
    *,
    min_train_days: int = 90,
    n_weeks: int = 2,
    n_paths: int = 200,
    horizon_days: int = 5,
    seed: int = 42,
    half_life_days: float = 2.0,
) -> dict:
    """A config.yaml-shaped mapping, scaled down so a walk-forward fits in a unit test.

    The shipped values (252 training days, 26 weeks, 5,000 paths) would mean hundreds of MLE fits
    per test. Only the sizes change here; every code path is the production one.
    """
    return {
        "catalog": "market_intel",
        "forecast": {"horizon_days": horizon_days, "n_paths": n_paths, "seed": seed},
        "news": {"half_life_days": half_life_days},
        "backtest": {
            "min_train_days": min_train_days,
            "origin_freq": "weekly",
            "n_weeks": n_weeks,
        },
    }


@pytest.fixture(scope="session")
def two_regime_returns_pct():
    """Percent log returns of the known two-regime series."""
    returns, _ = two_regime_series()
    return returns


@pytest.fixture(scope="session")
def true_regime_states():
    """The regime path that generated :func:`two_regime_returns_pct`."""
    _, states = two_regime_series()
    return states


@pytest.fixture(scope="session")
def regime_news(true_regime_states):
    """News signal aligned one-for-one with the return series, before any lagging."""
    return regime_news_series(true_regime_states)


@pytest.fixture(scope="session")
def fitted_markov(two_regime_returns_pct):
    """Model B fitted on the known series. Session-scoped: the MLE is the slow part."""
    from src.models.markov import fit_markov

    return fit_markov(two_regime_returns_pct)


@pytest.fixture(scope="session")
def sorted_markov(fitted_markov):
    """Model B's parameters after the mandatory re-sort."""
    from src.models.markov import sort_regimes

    return sort_regimes(fitted_markov)


@pytest.fixture(scope="session")
def fitted_news_markov(two_regime_returns_pct, regime_news):
    """Model C fitted on the same window, with the lagged news transition input."""
    from src.models.news_markov import fit_news_markov

    return fit_news_markov(two_regime_returns_pct, regime_news)


@pytest.fixture(scope="session")
def sorted_news_markov(fitted_news_markov):
    """Model C's parameters after the mandatory re-sort."""
    from src.models.markov import sort_regimes

    return sort_regimes(fitted_news_markov)


@pytest.fixture(scope="session")
def backtest_frame():
    """141 sessions of ``silver.daily_features`` for one ticker (spec B-5 input).

    Long enough for a 90-row training minimum plus several weekly origins, short enough that the
    MLE runs in a test. Session-scoped, so a test that needs to corrupt it must copy it first.
    """
    returns, states = two_regime_series(length=140)
    return features_frame(returns, regime_news_series(states))


@pytest.fixture(scope="session")
def backtest_cfg():
    """Scaled-down config for the walk-forward tests."""
    return backtest_config()


@pytest.fixture(scope="session")
def single_window(backtest_frame, backtest_cfg):
    """The most recent eligible origin's window: the object all three arms share."""
    from src.models.backtest import feature_dates, origin_window, weekly_origins

    dates = feature_dates(backtest_frame)
    origins = weekly_origins(
        backtest_frame,
        dates=dates,
        n_weeks=backtest_cfg["backtest"]["n_weeks"],
        min_train_days=60,
        horizon_days=backtest_cfg["forecast"]["horizon_days"],
    )
    return origin_window(backtest_frame, origins[-1], dates=dates, horizon_days=5)


# ------------------------------------------------------------------ fake SparkSession
# Shared by the pipeline tests (A-3/A-4) and the gold write tests (B-6). It lives here rather
# than in one test module because two copies of a fake are two contracts that drift apart, and
# the fake IS the contract for every ledgered write path in the repo.


class FakeResult:
    def __init__(self, row: dict | None = None, rows: list | None = None):
        self._row = row
        self._rows = rows or []

    def first(self):
        return self._row

    def collect(self):
        return self._rows


class FakeFrame:
    def __init__(self, spark: "FakeSpark", rows: list, schema: str):
        self._spark = spark
        self.rows = rows
        self.schema = schema

    def createOrReplaceTempView(self, name: str) -> None:  # noqa: N802 — Spark's API
        self._spark.views[name] = self


class FakeCatalog:
    def __init__(self, spark: "FakeSpark"):
        self._spark = spark

    def tableExists(self, fqn: str) -> bool:  # noqa: N802 — Spark's API
        return fqn not in self._spark.missing_tables

    def dropTempView(self, name: str) -> None:  # noqa: N802 — Spark's API
        self._spark.views.pop(name, None)


class FakeConf:
    def __init__(self):
        self.values: dict[str, str] = {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


class FakeSpark:
    """Enough of a SparkSession to run a task's ``main`` and inspect what it wrote.

    This is not a substitute for running against Delta — the SQL is never parsed — but the ledger
    contract and the MERGE-versus-INSERT contract are Python control flow, not SQL, and that is
    what these tests pin: one row per call, on success and on failure, with the right task name,
    and every write reaching Delta as a MERGE on the declared keys.
    """

    def __init__(self, *, row_count: int = 7, fail_on: str | None = None):
        self.catalog = FakeCatalog(self)
        self.conf = FakeConf()
        self.views: dict[str, FakeFrame] = {}
        self.missing_tables: set[str] = set()
        self.statements: list[str] = []
        self.frames: list[FakeFrame] = []
        self._row_count = row_count
        self._fail_on = fail_on

    def sql(self, text: str, args: dict | None = None) -> FakeResult:
        from datetime import date

        self.statements.append(text)
        if self._fail_on and self._fail_on in text:
            raise RuntimeError("boom: simulated Spark failure")
        if "min(trade_date)" in text:
            return FakeResult({"first_date": date(2026, 8, 3), "last_date": date(2026, 8, 14)})
        if "count(*) AS n FROM" in text:
            return FakeResult({"n": self._row_count})
        return FakeResult()

    def createDataFrame(self, rows, schema: str) -> FakeFrame:  # noqa: N802 — Spark's API
        frame = FakeFrame(self, list(rows), schema)
        self.frames.append(frame)
        return frame

    def ledger_rows(self) -> list[dict]:
        from src.ingestion import RUNS_COLUMNS, RUNS_SCHEMA_DDL

        return [
            dict(zip(RUNS_COLUMNS, row))
            for frame in self.frames
            if frame.schema == RUNS_SCHEMA_DDL
            for row in frame.rows
        ]

    def merge_statements(self) -> list[str]:
        return [text for text in self.statements if text.lstrip().startswith("MERGE INTO")]
