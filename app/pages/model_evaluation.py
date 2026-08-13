"""Model Evaluation page (spec A2, C-5): does the news-aware model actually beat the simpler ones?

Reads the pooled rows of ``gold.backtest_summary`` — one per arm — and renders them with n and the
fallback rate ALWAYS visible, plus one verdict line computed from the numbers.

THREE VERDICTS, AND THE BORING ONE IS FIRST-CLASS. Better, worse, or indistinguishable at this
sample size. The third is the outcome this project expects and it is stated as a finding rather
than hidden behind a chart that implies the winner: 130 forecasts is a small sample for a Brier
difference, and a page that always declares a champion is a page that will declare one from noise.

THE THRESHOLD. A Brier score is a mean of squared errors on probabilities, each bounded in [0, 1].
Under a null of "no real difference", a difference of pooled means of that size has a standard
error near ``sqrt(0.25 * 0.75 / n)`` — the spec's heuristic, taken from the variance of a
coin-flip-calibrated forecast. A spread inside that band is not evidence. This is a HEURISTIC and
:func:`verdict` says so on the page: it is not a paired test, it ignores the correlation between
arms that share an origin, and the arms are not independent samples. It is the right size of
instrument for a demo that must not overclaim, and calling it a p-value would be the overclaim.

FALLBACK RATE IS READ BEFORE THE SCORES. A news_markov column with a 40% fallback rate is not a
news_markov result — 40% of those rows were produced by a lower rung after a failed fit. The table
puts it in the same row as the score, and :func:`verdict` refuses to name a champion whose numbers
are mostly someone else's.

Every function above the reads is pure and is tested directly, with no Streamlit and no warehouse.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402 — the path has to be set before the app imports

from app.common import (  # noqa: E402
    CACHE_TTL_SECONDS,
    MISSING,
    WAREHOUSE_HINT,
    age_phrase,
    as_datetime,
    catalog,
    number,
    pct,
)
from src.database import delta  # noqa: E402

TITLE = "Model Evaluation"

SUMMARY_TABLE = "gold.backtest_summary"

#: The challenger and the baseline of the comparison the page exists to make. Both are arms of
#: ``gold.backtest_summary.model``; ``markov`` sits between them and is shown but not judged.
CHALLENGER = "news_markov"
BASELINE = "gbm"

#: Display order, richest model first. Anything unexpected in the table is appended rather than
#: dropped — an arm that appears in gold and not here is still evidence.
MODEL_ORDER = ("news_markov", "markov", "gbm")

MODEL_LABELS = {
    "news_markov": "News-Markov (Model C)",
    "markov": "Markov (Model B)",
    "gbm": "GBM (Model A)",
}

#: Above this, the arm's scores mostly describe a lower rung and the verdict says so instead of
#: crowning it. One in four is already generous.
FALLBACK_CEILING = 0.25

BETTER = "better"
WORSE = "worse"
INDISTINGUISHABLE = "indistinguishable"


# ------------------------------------------------------------------ pure functions


@dataclass(frozen=True)
class Verdict:
    """The verdict line, plus every number behind it so the page can show its work."""

    case: str
    text: str
    n: int = 0
    spread: float | None = None
    threshold: float | None = None
    challenger: str | None = None
    baseline: str | None = None


def threshold_for(n: int) -> float:
    """``sqrt(0.25 * 0.75 / n)`` — the spec's heuristic band for a Brier difference.

    Shrinks with the square root of the sample, so doubling n only narrows it by 40%: the reason a
    small backtest cannot resolve a small improvement, expressed as one number.
    """
    if n <= 0:
        raise ValueError("n must be positive to compute a threshold")
    return math.sqrt(0.25 * 0.75 / n)


def by_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("model")): row for row in rows if row.get("model")}


def ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows in display order, with anything unrecognized kept at the end."""
    indexed = by_model(rows)
    ordered = [indexed[model] for model in MODEL_ORDER if model in indexed]
    ordered.extend(row for row in rows if str(row.get("model")) not in MODEL_ORDER)
    return ordered


def verdict(rows: Sequence[Mapping[str, Any]]) -> Verdict:
    """The one sentence this page is for. Computed from the numbers, never from a preference.

    Compares the challenger's pooled Brier score against the baseline's. Lower is better, so the
    spread is ``baseline - challenger`` and a positive spread favours the challenger.
    """
    indexed = by_model(rows)
    challenger = indexed.get(CHALLENGER)
    baseline = indexed.get(BASELINE)

    if challenger is None or baseline is None:
        missing = [name for name in (CHALLENGER, BASELINE) if name not in indexed]
        return Verdict(
            case=INDISTINGUISHABLE,
            text=(
                f"No verdict: `{SUMMARY_TABLE}` has no pooled row for {', '.join(missing)}. "
                "Run the backtest notebook before reading this page."
            ),
        )

    n = min(int(challenger.get("n") or 0), int(baseline.get("n") or 0))
    if n <= 0:
        return Verdict(
            case=INDISTINGUISHABLE,
            text="No verdict: the pooled rows report n = 0, so nothing was actually scored.",
            challenger=CHALLENGER,
            baseline=BASELINE,
        )

    challenger_brier = _as_float(challenger.get("brier"))
    baseline_brier = _as_float(baseline.get("brier"))
    if challenger_brier is None or baseline_brier is None:
        return Verdict(
            case=INDISTINGUISHABLE,
            text="No verdict: a pooled Brier score is missing.",
            n=n,
            challenger=CHALLENGER,
            baseline=BASELINE,
        )

    spread = baseline_brier - challenger_brier
    threshold = threshold_for(n)
    fallback = _as_float(challenger.get("fallback_rate")) or 0.0

    detail = (
        f"Brier {challenger_brier:.4f} vs {baseline_brier:.4f} over n = {n} forecasts; "
        f"the difference is {abs(spread):.4f} against a noise band of ±{threshold:.4f} "
        f"(±sqrt(0.25 × 0.75 / n), a heuristic, not a significance test)."
    )

    if abs(spread) <= threshold:
        return Verdict(
            case=INDISTINGUISHABLE,
            text=(
                f"No meaningful improvement detected at this sample size. {detail} "
                "At this n the honest conclusion is that the two are indistinguishable."
            ),
            n=n,
            spread=spread,
            threshold=threshold,
            challenger=CHALLENGER,
            baseline=BASELINE,
        )

    if spread > 0:
        caveat = (
            ""
            if fallback <= FALLBACK_CEILING
            else (
                f" Read this with the fallback rate: {pct(fallback, 0)} of the "
                f"{MODEL_LABELS.get(CHALLENGER, CHALLENGER)} rows were produced by a simpler rung "
                "after a failed fit, so the score is not purely that model's."
            )
        )
        return Verdict(
            case=BETTER,
            text=f"{MODEL_LABELS[CHALLENGER]} scores better than {MODEL_LABELS[BASELINE]}. {detail}{caveat}",
            n=n,
            spread=spread,
            threshold=threshold,
            challenger=CHALLENGER,
            baseline=BASELINE,
        )

    return Verdict(
        case=WORSE,
        text=(
            f"{MODEL_LABELS[CHALLENGER]} scores WORSE than {MODEL_LABELS[BASELINE]}. {detail} "
            "The added complexity is not paying for itself here."
        ),
        n=n,
        spread=spread,
        threshold=threshold,
        challenger=CHALLENGER,
        baseline=BASELINE,
    )


def staleness_line(rows: Sequence[Mapping[str, Any]], now: datetime | None = None) -> str:
    """When these numbers were computed. The backtest is on demand, so this can be old."""
    stamps = [as_datetime(row.get("computed_at")) for row in rows]
    stamps = [stamp for stamp in stamps if stamp is not None]
    if not stamps:
        return f"`{SUMMARY_TABLE}` carries no computed_at, so the age of these numbers is unknown."

    newest = max(stamps)
    line = (
        f"Computed {age_phrase(newest, now)} ({newest.strftime('%Y-%m-%d %H:%M UTC')}). "
        "The backtest is an on-demand job, not part of the daily workflow."
    )
    if len(set(stamps)) > 1:
        line += " Rows carry different timestamps — some are from an earlier run."
    return line


def table_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The display table. n and fallback rate are columns, never a footnote (spec A2)."""
    return [
        {
            "Model": MODEL_LABELS.get(str(row.get("model")), str(row.get("model"))),
            "n": number(row.get("n")),
            "Tickers": number(row.get("n_tickers")),
            "Brier": _fixed(row.get("brier"), 4),
            "MAE": pct(row.get("mae"), 2),
            "80% coverage": pct(row.get("coverage_80"), 1),
            "Fallback rate": pct(row.get("fallback_rate"), 1),
        }
        for row in ordered_rows(rows)
    ]


def _fixed(value: Any, digits: int) -> str:
    number_value = _as_float(value)
    return MISSING if number_value is None else f"{number_value:.{digits}f}"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


# ------------------------------------------------------------------ reads


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def pooled_summary() -> list[dict]:
    return delta.query(f"SELECT * FROM {delta.qualified(catalog(), SUMMARY_TABLE)}")


# ------------------------------------------------------------------ rendering


def render() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":balance_scale:", layout="wide")
    st.title(TITLE)

    try:
        rows = pooled_summary()
    except Exception as exc:  # noqa: BLE001 — the diagnosis is known; show it, not a traceback
        st.error(WAREHOUSE_HINT)
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    if not rows:
        st.warning(
            f"`{SUMMARY_TABLE}` is empty. Run `notebooks/10_backtest_run.py` — the walk-forward "
            "backtest is an on-demand job, deliberately outside the daily workflow."
        )
        return

    result = verdict(rows)
    renderer = {BETTER: st.success, WORSE: st.error}.get(result.case, st.info)
    renderer(result.text)

    st.dataframe(table_rows(rows), hide_index=True, use_container_width=True)
    st.caption(staleness_line(rows))

    st.markdown(
        """
**How to read this table.** Brier scores P(return > 0) against what happened: 0 is perfect and
0.25 is an honest coin flip, so lower is better. MAE is the median forecast return's average
absolute error. 80% coverage should sit near 80% from either side — over-coverage means the
intervals are too wide, which is as much a failure as too narrow. The fallback rate is how often
the arm's fit failed and a simpler model answered in its place; read it before the scores.
"""
    )


if __name__ == "__main__":
    render()
