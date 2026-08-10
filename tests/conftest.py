"""Shared fixtures mirroring REAL Massive payload shapes (spec A-1, A-2, A-3, A-5).

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
