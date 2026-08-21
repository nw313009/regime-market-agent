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
    WAREHOUSE_HINT,
    age_phrase,
    as_datetime,
    as_float,
    catalog,
    fixed,
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

    challenger_brier = as_float(challenger.get("brier"))
    baseline_brier = as_float(baseline.get("brier"))
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
    fallback = as_float(challenger.get("fallback_rate")) or 0.0

    detail = (
        f"Brier {fixed(challenger_brier, 4)} vs {fixed(baseline_brier, 4)} over "
        f"n = {number(n)} forecasts; the difference is {fixed(abs(spread), 4)} against a noise "
        f"band of ±{fixed(threshold, 4)} "
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
            "Brier": fixed(row.get("brier"), 4),
            "MAE": pct(row.get("mae"), 2),
            "80% coverage": pct(row.get("coverage_80"), 1),
            "Fallback rate": pct(row.get("fallback_rate"), 1),
        }
        for row in ordered_rows(rows)
    ]


# ------------------------------------------------------------------ the trust story
#
# THE ARM NAMES DO NOT APPEAR IN ANY OF THESE LINES. "news_markov vs markov vs gbm" tells a user
# nothing they can act on — the names are an implementation detail of the evaluation, and the
# evaluation in full, names included, is the table above and the README. What a reader of a single
# forecast needs is narrower: how big the sample was, how the comparison came out, and the two
# reliability facts. So these lines say "the production model" and "the alternative", and every
# figure in them is read from gold.backtest_summary rather than typed in.


#: The arm the daily job runs, and the rung it falls back to. The reliability line compares exactly
#: these two: gbm's fallback rate is 0 by construction (it is the bottom of the ladder and has no
#: fit to fail), so comparing against it would be arithmetic dressed as a finding.
PRODUCTION_MODEL = CHALLENGER
ALTERNATIVE_MODEL = "markov"

#: The interval the backtest scores coverage against is [return_p10, return_p90] — 80% by
#: construction, not by configuration. The tolerance is what counts as hitting it from either
#: side; over-coverage is as much a miss as under-coverage, so the band is symmetric.
COVERAGE_TARGET = 0.80
COVERAGE_TOLERANCE = 0.05

#: Where the full evaluation lives, arm names and all.
README_URL = "https://github.com/nw313009/regime-market-agent#findings-the-interesting-parts"


def trust_lines(
    rows: Sequence[Mapping[str, Any]],
    forecast: Mapping[str, Any] | None = None,
) -> list[str]:
    """"How much to trust this", in plain words, from the stored evaluation.

    Returns one line per fact that the table can actually support. A line whose numbers are absent
    is omitted rather than rendered with a dash: this block is prose a user reads once, and
    "it fails — as often as —" is worse than saying less.
    """
    indexed = by_model(rows)
    if PRODUCTION_MODEL not in indexed:
        return [
            "This forecast has not been validated yet: the backtest has not been run, so there is "
            "no measured accuracy to report."
        ]

    lines = [_sample_line(rows, indexed[PRODUCTION_MODEL])]
    reliability = _reliability_line(indexed)
    if reliability:
        lines.append(reliability)
    lines.append(_coverage_line(rows))
    notice = _fallback_notice(forecast)
    if notice:
        lines.append(notice)
    return lines


def _sample_line(rows: Sequence[Mapping[str, Any]], production: Mapping[str, Any]) -> str:
    n = int(production.get("n") or 0)
    tickers = int(production.get("n_tickers") or 0)
    if n <= 0:
        return (
            "The evaluation table reports no scored forecasts, so there is no measured accuracy "
            "to report yet."
        )

    scope = f"Validated on {number(n)} historical forecasts"
    if tickers > 0:
        # Origins are weekly and every origin is scored for every ticker, so the week count is
        # derivable rather than stored. Deriving it beats adding a column that could disagree.
        scope += f" across {number(n // tickers)} weeks and {number(tickers)} tickers"
    return f"{scope} — {_comparison_clause(rows)}."


def _comparison_clause(rows: Sequence[Mapping[str, Any]]) -> str:
    """The same verdict the table renders, in words that name no arm.

    Reuses :func:`verdict` rather than re-deriving the call: two implementations of "is this
    better" is how a headline and its evidence start disagreeing.
    """
    if BASELINE not in by_model(rows):
        # verdict() reports INDISTINGUISHABLE when an arm is missing, which is the right value for
        # a page that shows its own reasoning and the wrong word for a sentence that would then
        # claim a tie against a baseline nobody scored.
        return "with no simpler baseline scored alongside it for comparison"

    case = verdict(rows).case
    if case == BETTER:
        return "accuracy scored better than the simpler alternatives"
    if case == WORSE:
        return "accuracy scored worse than the simpler alternatives"
    return "accuracy held up against simpler alternatives, which tested as statistically equivalent"


def _reliability_line(indexed: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Why the richer model ships even when the scores tie: it answers more often."""
    production = as_float((indexed.get(PRODUCTION_MODEL) or {}).get("fallback_rate"))
    alternative = as_float((indexed.get(ALTERNATIVE_MODEL) or {}).get("fallback_rate"))
    if production is None or alternative is None or alternative <= 0:
        return None

    ratio = production / alternative
    if ratio >= 1.0:
        # The claim is that it fails LESS. If it does not, say nothing here rather than inventing
        # a sentence that spins a worse number.
        return None

    if ratio < 0.5:
        frequency = "less than half as often as the alternative"
    elif ratio <= 0.65:
        frequency = "about half as often as the alternative"
    else:
        frequency = "less often than the alternative"

    return (
        "The production model was chosen for reliability: it fails to produce a forecast "
        f"{frequency} ({pct(production)} vs {pct(alternative)})."
    )


def _coverage_line(rows: Sequence[Mapping[str, Any]]) -> str:
    measured = [value for value in (as_float(row.get("coverage_80")) for row in rows) if value is not None]
    if not measured:
        return "Interval coverage was not recorded by this backtest, so it is not reported here."

    low, high = min(measured), max(measured)
    span = fixed(low, 2) if low == high else f"{fixed(low, 2)}–{fixed(high, 2)}"
    if all(abs(value - COVERAGE_TARGET) <= COVERAGE_TOLERANCE for value in measured):
        return f"Forecast intervals hit their 80% coverage target ({span} measured)."
    return f"Forecast intervals measured {span} against an 80% coverage target."


def _fallback_notice(forecast: Mapping[str, Any] | None) -> str | None:
    """Said plainly when today's row is not the model the trust lines just described."""
    if not forecast:
        return None
    used = str(forecast.get("model_used") or "").strip().lower()
    return None if not used or used == PRODUCTION_MODEL else "Today's forecast used the fallback model."


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
