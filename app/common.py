"""Shared plumbing for the three pages (spec C-5): paths, config, formatting, disclosures.

NO PAGE CONTENT LIVES HERE and no query does either — each page owns its own reads, so the SQL
that fills a card sits next to the card. What is shared is what would otherwise be copied four
times: putting the repo root on ``sys.path``, reading config once, turning a decimal into a
percentage the same way everywhere, and the two sentences the product is obliged to say.

EVERY FUNCTION HERE IS PURE except :func:`config`, which reads a file. That is deliberate: the
tests import the page modules and call the formatting and verdict logic directly, with no
Streamlit server and no warehouse, and none of it can be pure if it is entangled with ``st``.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    """Put the repo root on ``sys.path`` so ``src.*`` and ``app.*`` imports resolve.

    Streamlit puts the MAIN SCRIPT'S directory on the path — ``app/``, not the repo root — so
    without this a page finds ``common`` and not ``src.agent``. Same shape as the notebook wrapper
    (spec C-a); the app has the same problem for the same reason.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


ensure_repo_on_path()

from src.llm import load_config  # noqa: E402 — the path has to be set first

#: Cache TTL for every warehouse read (spec C-5). Ten minutes: the data behind these pages changes
#: once a day, and a demo that re-queries on every widget interaction spends its time waiting on a
#: warehouse instead of showing the product.
CACHE_TTL_SECONDS = 600

#: What to say when the warehouse refuses. Named here because the diagnosis is always the same
#: (spec C-e) and a stack trace on a demo screen teaches the audience nothing.
WAREHOUSE_HINT = (
    "Could not read Delta. This is almost always the warehouse id or its permissions in "
    "`app.yaml` rather than a bug in the page — check `DATABRICKS_WAREHOUSE_ID` and that the "
    "app's service principal can use the warehouse and the `market_intel` catalog."
)

LAKEBASE_HINT = (
    "Could not reach Lakebase. Check `LAKEBASE_ENDPOINT` / `PGHOST` / `PGUSER` in `app.yaml` and "
    "that the app's service principal has a Postgres role."
)


def config() -> Mapping[str, Any]:
    """The repository config. One read per process; :func:`load_config` caches it."""
    return load_config()


def catalog(cfg: Mapping[str, Any] | None = None) -> str:
    return str((cfg or config())["catalog"])


def seed_tickers(cfg: Mapping[str, Any] | None = None) -> list[str]:
    source = cfg or config()
    return sorted({str(t).strip().upper() for t in (source.get("tickers") or {}).get("seed") or []})


def half_life_days(cfg: Mapping[str, Any] | None = None) -> float:
    source = cfg or config()
    return float((source.get("news") or {}).get("half_life_days", 2))


def decay_disclosure(half_life: float | None = None) -> str:
    """THE DISCLOSURE SENTENCE (spec A2). Rendered on the forecast, not tucked into a footer.

    It exists because the model conditions on TODAY's news and decays it — it does not predict
    future news — and that is an assumption of the model rather than something measured from the
    data. A forecast shown without it invites the reader to believe the system knows what will be
    published tomorrow.
    """
    life = half_life_days() if half_life is None else float(half_life)
    return (
        f"News assumption: the forecast conditions on the CURRENT news sentiment and decays it "
        f"over the horizon with a {_number(life)}-trading-day half-life. It does not predict "
        f"future news."
    )


def _number(value: float) -> str:
    """``2.0`` reads as ``2``; ``1.5`` stays ``1.5``."""
    return str(int(value)) if float(value).is_integer() else str(value)


# ------------------------------------------------------------------ formatting


MISSING = "—"


def pct(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    """A DECIMAL fraction as a percentage. Every number in gold is decimal (spec B-6)."""
    if value is None:
        return MISSING
    number = float(value) * 100.0
    if number != number:  # NaN
        return MISSING
    return f"{number:+.{digits}f}%" if signed else f"{number:.{digits}f}%"


def money(value: Any, digits: int = 2) -> str:
    if value is None:
        return MISSING
    number = float(value)
    return MISSING if number != number else f"${number:,.{digits}f}"


def number(value: Any, digits: int = 0) -> str:
    if value is None:
        return MISSING
    return f"{float(value):,.{digits}f}"


def as_datetime(value: Any) -> datetime | None:
    """Normalize whatever the warehouse hands back for a TIMESTAMP to an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return as_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    converted = getattr(value, "to_pydatetime", None)
    return as_datetime(converted()) if converted is not None else None


def age_phrase(moment: Any, now: datetime | None = None) -> str:
    """"14 minutes ago" / "3 days ago". PURE, and tested directly.

    Whole units only. A staleness line is read at a glance to answer one question — is this from
    today — and "2 days, 4:31:07" makes that question harder rather than easier.
    """
    when = as_datetime(moment)
    if when is None:
        return "at an unknown time"

    reference = now or datetime.now(timezone.utc)
    seconds = (reference - when).total_seconds()
    if seconds < 0:
        return "just now"

    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        count = int(seconds // size)
        if count >= 1:
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    return "less than a minute ago"
