"""Streamlit app tests (spec C-5).

WHAT THESE TESTS CAN AND CANNOT DO. There is no Streamlit server here and no warehouse, so nothing
below renders a page. What they pin is everything that would otherwise only be checked by loading
the app in a browser and squinting:

1. THE PAGES IMPORT CLEAN. Importing a page module must execute no Streamlit call and render
   nothing. That is what makes the pure functions testable at all, and it is enforced twice — the
   import itself, and an AST check that no top-level statement calls ``st.*`` or ``render()``
   outside the ``__main__`` guard. Streamlit execs a page under the name ``__main__``, so the
   guard is not a Python-only convention here: it is what the runtime actually triggers.
2. THE VERDICT LINE, in all three cases. This is the sentence the Model Evaluation page exists to
   produce and the one most likely to overclaim, so better / worse / indistinguishable are each
   asserted against numbers chosen to sit clearly on their side of the threshold.
3. THE STALENESS LINE, including the case where the timestamp is missing entirely.
4. app.yaml carries the environment the app cannot start without, and carries no password.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from app import common
from app.pages import market_research, model_evaluation, research_agent

REPO_ROOT = Path(__file__).resolve().parents[1]

PAGE_MODULES = (
    "app.app",
    "app.pages.market_research",
    "app.pages.model_evaluation",
    "app.pages.research_agent",
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def pooled(model: str, brier: float, n: int = 130, **overrides) -> dict:
    """One ``gold.backtest_summary`` row, with the columns the page actually reads."""
    row = {
        "model": model,
        "n": n,
        "n_tickers": 5,
        "brier": brier,
        "mae": 0.031,
        "coverage_80": 0.79,
        "fallback_rate": 0.0,
        "computed_at": NOW - timedelta(hours=3),
    }
    row.update(overrides)
    return row


class TestPagesImportClean:
    """A page must be importable without a Streamlit server and without rendering anything."""

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_the_module_imports(self, module_name):
        module = importlib.import_module(module_name)
        assert callable(module.render), f"{module_name} must expose render()"

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_nothing_renders_at_import_time(self, module_name):
        """No top-level ``st.*`` call and no bare ``render()`` outside the ``__main__`` guard.

        The import above would not catch this: Streamlit calls outside a script run only WARN, so
        a page that rendered at import would pass a smoke import and then draw itself twice in the
        app. The AST is what makes the absence provable.
        """
        module = importlib.import_module(module_name)
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

        offenders = []
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value.func
            name = getattr(call, "attr", None) or getattr(call, "id", None)
            if isinstance(call, ast.Attribute) and getattr(call.value, "id", "") == "st":
                offenders.append(f"st.{name}")
            elif name == "render":
                offenders.append("render")

        assert not offenders, (
            f"{module_name} calls {offenders} at import time. Rendering belongs under "
            "`if __name__ == \"__main__\": render()` — Streamlit execs a page as __main__, so the "
            "guard still runs in the app."
        )

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_rendering_is_behind_the_main_guard(self, module_name):
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source
        assert source.rstrip().endswith("render()")


class TestVerdict:
    """Three cases, and the boring one is not a fallback for a broken input."""

    def test_a_clear_improvement_reads_as_better(self):
        rows = [pooled("news_markov", 0.180), pooled("gbm", 0.250)]

        result = model_evaluation.verdict(rows)

        assert result.case == model_evaluation.BETTER
        assert result.spread == pytest.approx(0.070)
        assert result.spread > result.threshold
        assert "better" in result.text.lower()

    def test_a_clear_regression_reads_as_worse(self):
        rows = [pooled("news_markov", 0.250), pooled("gbm", 0.180)]

        result = model_evaluation.verdict(rows)

        assert result.case == model_evaluation.WORSE
        assert result.spread == pytest.approx(-0.070)
        assert "WORSE" in result.text

    def test_a_small_spread_at_this_n_is_indistinguishable(self):
        """The first-class outcome (spec A2), not an error path."""
        rows = [pooled("news_markov", 0.2470), pooled("gbm", 0.2500)]

        result = model_evaluation.verdict(rows)

        assert result.case == model_evaluation.INDISTINGUISHABLE
        assert abs(result.spread) < result.threshold
        assert "No meaningful improvement detected at this sample size" in result.text
        assert "heuristic" in result.text

    def test_the_threshold_is_the_specified_heuristic(self):
        assert model_evaluation.threshold_for(130) == pytest.approx((0.25 * 0.75 / 130) ** 0.5)
        assert model_evaluation.threshold_for(520) == pytest.approx(
            model_evaluation.threshold_for(130) / 2
        )
        with pytest.raises(ValueError):
            model_evaluation.threshold_for(0)

    def test_the_same_spread_flips_the_verdict_as_n_grows(self):
        """The point of the band: evidence is a spread RELATIVE to the sample size."""
        small = model_evaluation.verdict(
            [pooled("news_markov", 0.220, n=30), pooled("gbm", 0.250, n=30)]
        )
        large = model_evaluation.verdict(
            [pooled("news_markov", 0.220, n=3000), pooled("gbm", 0.250, n=3000)]
        )

        assert small.case == model_evaluation.INDISTINGUISHABLE
        assert large.case == model_evaluation.BETTER

    def test_n_is_the_smaller_of_the_two_arms(self):
        result = model_evaluation.verdict(
            [pooled("news_markov", 0.20, n=90), pooled("gbm", 0.25, n=130)]
        )
        assert result.n == 90

    def test_a_champion_built_on_fallbacks_is_flagged(self):
        rows = [pooled("news_markov", 0.180, fallback_rate=0.4), pooled("gbm", 0.250)]

        result = model_evaluation.verdict(rows)

        assert result.case == model_evaluation.BETTER
        assert "fallback rate" in result.text
        assert "40%" in result.text

    def test_a_missing_arm_produces_no_verdict_rather_than_a_guess(self):
        result = model_evaluation.verdict([pooled("markov", 0.21)])

        assert result.case == model_evaluation.INDISTINGUISHABLE
        assert "No verdict" in result.text
        assert "news_markov" in result.text and "gbm" in result.text

    def test_zero_scored_forecasts_produce_no_verdict(self):
        rows = [pooled("news_markov", 0.0, n=0), pooled("gbm", 0.0, n=0)]

        result = model_evaluation.verdict(rows)

        assert result.case == model_evaluation.INDISTINGUISHABLE
        assert "n = 0" in result.text


class TestEvaluationTable:
    def test_n_and_fallback_rate_are_always_columns(self):
        """Spec A2 makes both mandatory on the page; a missing value shows as a dash, not a gap."""
        rows = model_evaluation.table_rows([pooled("gbm", 0.25, fallback_rate=None, n=130)])

        assert rows[0]["n"] == "130"
        assert rows[0]["Fallback rate"] == common.MISSING

    def test_rows_are_ordered_richest_model_first(self):
        rows = model_evaluation.table_rows(
            [pooled("gbm", 0.25), pooled("news_markov", 0.21), pooled("markov", 0.23)]
        )
        assert [row["Model"] for row in rows] == [
            model_evaluation.MODEL_LABELS["news_markov"],
            model_evaluation.MODEL_LABELS["markov"],
            model_evaluation.MODEL_LABELS["gbm"],
        ]

    def test_an_unknown_arm_is_kept_rather_than_dropped(self):
        rows = model_evaluation.table_rows([pooled("something_new", 0.2)])
        assert rows[0]["Model"] == "something_new"


class TestStaleness:
    def test_it_reports_the_newest_timestamp_and_its_age(self):
        line = model_evaluation.staleness_line([pooled("gbm", 0.25)], now=NOW)

        assert "3 hours ago" in line
        assert "2026-08-12 09:00 UTC" in line
        assert "on-demand" in line

    def test_mixed_timestamps_are_called_out(self):
        rows = [
            pooled("gbm", 0.25, computed_at=NOW - timedelta(hours=3)),
            pooled("news_markov", 0.21, computed_at=NOW - timedelta(days=9)),
        ]

        line = model_evaluation.staleness_line(rows, now=NOW)

        assert "3 hours ago" in line  # the newest, not the oldest
        assert "earlier run" in line

    def test_a_missing_timestamp_is_stated_rather_than_hidden(self):
        line = model_evaluation.staleness_line([pooled("gbm", 0.25, computed_at=None)], now=NOW)
        assert "unknown" in line

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (timedelta(seconds=20), "less than a minute ago"),
            (timedelta(minutes=1), "1 minute ago"),
            (timedelta(minutes=14), "14 minutes ago"),
            (timedelta(hours=1), "1 hour ago"),
            (timedelta(days=3), "3 days ago"),
            (timedelta(days=-1), "just now"),
        ],
    )
    def test_age_phrase_uses_whole_units(self, delta, expected):
        assert common.age_phrase(NOW - delta, now=NOW) == expected


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0312, "3.1%"), (-0.0312, "-3.1%"), (None, common.MISSING), (float("nan"), common.MISSING)],
    )
    def test_pct_reads_gold_decimals_as_percentages(self, value, expected):
        assert common.pct(value) == expected

    def test_pct_can_show_the_sign_for_returns(self):
        assert common.pct(0.0312, signed=True) == "+3.1%"

    def test_money_and_number(self):
        assert common.money(1234.5) == "$1,234.50"
        assert common.number(5000) == "5,000"
        assert common.money(None) == common.MISSING

    def test_fixed_is_the_one_way_a_bare_decimal_reaches_a_screen(self):
        """Brier scores and coverage rates. Inline f-strings are how one figure gets two roundings."""
        assert common.fixed(0.25301, 4) == "0.2530"
        assert common.fixed(0.816) == "0.82"
        assert common.fixed(None) == common.MISSING
        assert common.fixed(float("nan")) == common.MISSING

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, 1.0), ("2.5", 2.5), (None, None), ("abc", None), (float("nan"), None)],
    )
    def test_as_float_answers_is_there_a_number_here(self, value, expected):
        """NaN comes back None on purpose: a NaN that survives fails silently in a `<` comparison."""
        assert common.as_float(value) == expected

    def test_the_horizon_comes_from_config_like_the_half_life(self):
        assert common.horizon_days() == 5
        assert common.horizon_days({"forecast": {"horizon_days": 10}}) == 10
        assert common.horizon_days({"catalog": "market_intel"}) == 5

    def test_the_decay_disclosure_states_the_configured_half_life(self):
        """Interpolated from config, so changing the config cannot leave the page lying."""
        assert "2-trading-day half-life" in common.decay_disclosure()
        assert "does not predict future news" in common.decay_disclosure()
        assert "1-trading-day" in common.decay_disclosure(1)


class TestMarketResearchPage:
    def test_the_regime_headline_names_the_dominant_regime(self):
        assert market_research.regime_headline({"prob_low_vol": 0.27, "prob_high_vol": 0.73}) == (
            "High volatility — 73%"
        )
        assert market_research.regime_headline({"prob_low_vol": 0.81, "prob_high_vol": 0.19}) == (
            "Low volatility — 81%"
        )

    def test_a_missing_regime_row_says_so(self):
        assert "No regime estimate" in market_research.regime_headline(None)

    def test_a_gbm_day_has_no_regime_probabilities(self):
        """The daily task writes no regime row for a gbm fallback; the page must not show zeros."""
        row = {"model_used": "gbm", "prob_low_vol": None, "prob_high_vol": None}
        assert "no regimes" in market_research.regime_headline(row)

    def test_silence_and_indifference_are_different_news_states(self):
        empty = market_research.news_tone([])
        neutral = market_research.news_tone([{"sentiment_label": "neutral"}] * 3)

        assert "No news in the window" in empty
        assert "absence of news, not a negative view" in empty
        assert "3 articles in the window: 3 neutral." == neutral

    def test_the_tone_line_counts_every_label_including_unlabelled(self):
        rows = [
            {"sentiment_label": "positive"},
            {"sentiment_label": "positive"},
            {"sentiment_label": None},
        ]
        assert market_research.news_tone(rows) == (
            "3 articles in the window: 2 positive, 1 unlabelled."
        )

    def test_chart_data_is_columns_and_needs_no_pandas(self):
        rows = [{"trade_date": "2026-08-10", "close": 101.5}, {"trade_date": "2026-08-11", "close": None}]

        data = market_research.chart_data(rows)

        assert data == {"trade_date": ["2026-08-10", "2026-08-11"], "close": [101.5, None]}

    def test_the_forecast_caption_carries_its_provenance(self):
        caption = market_research.forecast_caption(
            {"model_used": "news_markov", "as_of_date": "2026-08-11", "n_paths": 5000, "horizon_days": 5}
        )
        assert "news_markov" in caption and "5,000 simulated paths" in caption

    def test_reads_are_cached(self):
        """``st.cache_data(ttl=600)`` on every warehouse read (spec C-5)."""
        assert common.CACHE_TTL_SECONDS == 600
        for reader in (
            market_research.price_history,
            market_research.current_regime,
            market_research.latest_forecast,
            market_research.recent_news,
            model_evaluation.pooled_summary,
        ):
            assert hasattr(reader, "clear"), f"{reader} is not wrapped in st.cache_data"


class TestResearchAgentPage:
    def test_the_question_carries_the_selected_ticker(self):
        assert research_agent.scoped_question("NVDA", "  why is risk up? ") == (
            "[Ticker: NVDA] why is risk up?"
        )

    def test_tool_activity_reads_as_plain_language(self, agent_result):
        assert research_agent.tool_activity(agent_result) == [
            "checked the forecast (NVDA)",
            "searched news (downside risk)",
        ]

    def test_a_failed_tool_call_is_shown_as_failed(self, agent_result):
        agent_result.tool_calls[1] = dataclasses.replace(agent_result.tool_calls[1], ok=False)
        assert "failed" in research_agent.tool_activity(agent_result)[1]

    def test_the_iteration_limit_is_visible_in_the_trail(self, agent_result):
        agent_result.hit_iteration_limit = True
        assert research_agent.tool_activity(agent_result)[-1] == "stopped at the tool-call limit"

    def test_no_result_means_no_trail(self):
        assert research_agent.tool_activity(None) == []

    def test_the_saved_report_points_at_the_forecast_the_turn_used(self, agent_result):
        assert research_agent.used_forecast_id(agent_result) == "forecast-123"

    def test_a_turn_that_never_read_a_forecast_saves_without_one(self, agent_result):
        agent_result.tool_calls = agent_result.tool_calls[1:]
        assert research_agent.used_forecast_id(agent_result) is None

    def test_every_tool_the_agent_has_gets_a_label(self):
        """A new tool without a label would show its function name to a demo audience."""
        from src.agent import tools as agent_tools

        assert {tool.name for tool in agent_tools.TOOLS} == set(research_agent.TOOL_LABELS)


#: The words no UI string may contain (spec: the app explains, it does not advise). Matched on word
#: BOUNDARIES, not substrings — "threshold" contains "hold" and "held up" contains neither, and a
#: naive `in` check would either fail on the first or miss nothing useful.
BANNED_UI_WORDS = ("consider", "opportunity", "entry", "buy", "sell", "hold")


def assert_no_advice_words(text: str) -> None:
    found = [word for word in BANNED_UI_WORDS if re.search(rf"\b{word}\w*\b", text, re.IGNORECASE)]
    assert not found, f"UI copy contains advice words {found}: {text!r}"


def forecast_row(**overrides) -> dict:
    """One ``gold.forecast_runs`` row, with the columns the redesigned cards read."""
    row = {
        "forecast_id": "forecast-123",
        "as_of_date": "2026-08-11",
        "horizon_days": 5,
        "model_used": "news_markov",
        "current_price": 100.0,
        "price_p10": 94.0,
        "price_p50": 100.5,
        "price_p90": 108.0,
        "return_p10": -0.060,
        "return_p50": 0.005,
        "return_p90": 0.080,
        "prob_positive": 0.540,
        "prob_loss_gt_5pct": 0.151,
        "prob_low_vol": 0.983,
        "prob_high_vol": 0.017,
        "n_paths": 5000,
    }
    row.update(overrides)
    return row


class TestHeadlineCard:
    """The first line a reader sees is assembled from bands, so the bands are what get pinned."""

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.10, market_research.LEANING_NEGATIVE),
            (0.449, market_research.LEANING_NEGATIVE),
            (0.45, market_research.COIN_FLIP),
            (0.50, market_research.COIN_FLIP),
            (0.55, market_research.COIN_FLIP),
            (0.551, market_research.LEANING_POSITIVE),
            (0.97, market_research.LEANING_POSITIVE),
        ],
    )
    def test_the_direction_phrase_cuts_at_the_declared_bands(self, probability, expected):
        assert market_research.direction_phrase(probability) == expected

    def test_a_fifty_four_percent_forecast_is_not_called_positive(self):
        """The reason the bands exist. 54% is a coin flip a reader could otherwise act on."""
        assert market_research.direction_phrase(0.54) == market_research.COIN_FLIP

    def test_a_missing_probability_has_no_phrase_rather_than_a_neutral_one(self):
        assert market_research.direction_phrase(None) is None
        assert market_research.direction_phrase(float("nan")) is None

    def test_the_regime_badge_names_the_dominant_regime_and_its_tone(self):
        calm, tone = market_research.regime_badge({"prob_low_vol": 0.983, "prob_high_vol": 0.017})
        assert calm == "Calm market · 98.3%"
        assert tone == market_research.CALM

        turbulent, tone = market_research.regime_badge({"prob_low_vol": 0.27, "prob_high_vol": 0.73})
        assert turbulent == "Turbulent market · 73.0%"
        assert tone == market_research.TURBULENT

    def test_a_gbm_day_gets_an_honest_badge_rather_than_zeros(self):
        badge, tone = market_research.regime_badge({"prob_low_vol": None, "prob_high_vol": None})

        assert badge == "Regime unknown"
        assert tone == market_research.UNKNOWN
        assert market_research.regime_badge(None)[1] == market_research.UNKNOWN

    def test_the_verdict_carries_the_band_and_the_regime(self):
        text = market_research.headline_verdict(forecast_row(), {"prob_low_vol": 0.98, "prob_high_vol": 0.02})

        assert market_research.COIN_FLIP in text
        assert "calm" in text
        assert "5 trading days" in text

    def test_the_regime_changes_the_colour_not_the_wording(self):
        """Only the tone word differs. A sentence that gets more alarming has an opinion."""
        calm = market_research.headline_verdict(forecast_row(), {"prob_low_vol": 0.9, "prob_high_vol": 0.1})
        turbulent = market_research.headline_verdict(forecast_row(), {"prob_low_vol": 0.1, "prob_high_vol": 0.9})

        assert calm.replace("calm", "X") == turbulent.replace("turbulent", "X")

    def test_it_falls_back_to_the_forecast_row_when_there_is_no_regime_row(self):
        """``forecast_runs`` carries the same two probabilities, so a missing regime row is not blank."""
        text = market_research.headline_verdict(forecast_row(), None)
        assert "calm" in text

    def test_a_gbm_day_says_the_regime_could_not_be_estimated(self):
        row = forecast_row(prob_low_vol=None, prob_high_vol=None)
        assert "could not estimate" in market_research.headline_verdict(row, None)

    def test_no_forecast_and_no_probability_each_say_so(self):
        assert "No forecast has been computed" in market_research.headline_verdict(None)
        assert "no direction probability" in market_research.headline_verdict(
            forecast_row(prob_positive=None)
        )

    def test_the_metric_label_takes_the_horizon_from_the_row(self):
        assert market_research.direction_label(forecast_row()) == "Positive 5-day return"
        assert market_research.direction_label(forecast_row(horizon_days=10)) == (
            "Positive 10-day return"
        )

    def test_the_horizon_falls_back_to_config_when_the_row_omits_it(self):
        assert market_research.forecast_horizon(forecast_row(horizon_days=None)) == 5
        assert market_research.forecast_horizon(None) == 5


class TestRangeCard:
    def test_each_percentile_carries_a_price_and_a_signed_return(self):
        rows = market_research.range_rows(forecast_row())

        assert [row["label"] for row in rows] == [
            "Low · 10th percentile",
            "Median",
            "High · 90th percentile",
        ]
        assert rows[0]["price"] == "$94.00"
        assert rows[0]["return"] == "-6.0%"
        assert rows[2]["return"] == "+8.0%"

    def test_no_forecast_produces_no_rows(self):
        assert market_research.range_rows(None) == []

    def test_the_band_marker_sits_where_the_median_falls(self):
        assert market_research.band_offset(
            forecast_row(price_p10=90.0, price_p50=100.0, price_p90=110.0)
        ) == pytest.approx(50.0)
        assert market_research.band_offset(
            forecast_row(price_p10=90.0, price_p50=92.0, price_p90=110.0)
        ) == pytest.approx(10.0)

    def test_a_degenerate_band_draws_no_bar_rather_than_a_pinned_marker(self):
        assert market_research.band_offset(forecast_row(price_p90=94.0)) is None
        assert market_research.band_offset(forecast_row(price_p50=None)) is None
        assert market_research.band_offset(None) is None

    def test_the_downside_line_is_the_plainest_wording_available(self):
        assert market_research.loss_line(forecast_row()) == (
            "Chance of losing more than 5%: 15.1%."
        )
        assert "not recorded" in market_research.loss_line(forecast_row(prob_loss_gt_5pct=None))

    def test_the_caption_says_what_the_three_numbers_are(self):
        assert market_research.distribution_caption(forecast_row()) == (
            "Distribution across 5,000 simulated paths at a 5-trading-day horizon."
        )

    def test_an_unrecorded_path_count_is_stated_rather_than_guessed(self):
        caption = market_research.distribution_caption(forecast_row(n_paths=None))
        assert "unrecorded number of simulated paths" in caption


class TestTrustStory:
    """Three lines a user can act on, from the stored evaluation, naming no model."""

    @staticmethod
    def evaluation(**overrides) -> list[dict]:
        """The frozen run: n=255 over 51 weekly origins, arms tied, fallback halved."""
        shared = {"n": 255, "n_tickers": 5, "coverage_80": 0.82} | overrides
        return [
            pooled("news_markov", 0.25301, fallback_rate=0.0745, **shared),
            pooled("markov", 0.25335, fallback_rate=0.1686, **shared),
            pooled("gbm", 0.25374, fallback_rate=0.0, **shared),
        ]

    def test_no_line_names_a_model(self):
        """The whole point. "news_markov vs markov vs gbm" belongs in the README, not the app."""
        lines = " ".join(model_evaluation.trust_lines(self.evaluation(), forecast_row()))

        for name in ("news_markov", "markov", "gbm", "Model A", "Model B", "Model C"):
            assert name.lower() not in lines.lower()

    def test_the_sample_line_derives_the_week_count_from_n(self):
        """255 forecasts over 5 tickers is 51 weekly origins — derived, never a second column."""
        line = model_evaluation.trust_lines(self.evaluation())[0]

        assert "255 historical forecasts" in line
        assert "51 weeks" in line
        assert "5 tickers" in line

    def test_a_tie_reads_as_statistically_equivalent(self):
        line = model_evaluation.trust_lines(self.evaluation())[0]
        assert "statistically equivalent" in line

    def test_a_real_improvement_is_not_reported_as_a_tie(self):
        rows = self.evaluation()
        rows[0] = pooled("news_markov", 0.15, n=255, n_tickers=5, fallback_rate=0.0745)

        assert "scored better" in model_evaluation.trust_lines(rows)[0]

    def test_the_reliability_line_compares_against_the_alternative_not_the_baseline(self):
        """gbm's fallback rate is 0 by construction; comparing to it would be arithmetic, not a finding.

        Asserted through :func:`common.pct` rather than against a typed-in "7.5%": the point of the
        test is WHICH arm is on the other side of the comparison, and pinning the rendering of
        0.0745 here would make this test fail the day the formatter's precision changes.
        """
        lines = model_evaluation.trust_lines(self.evaluation())
        reliability = next(line for line in lines if "reliability" in line)

        assert common.pct(0.0745) in reliability
        assert common.pct(0.1686) in reliability, "the alternative's rate, not the baseline's"
        assert common.pct(0.0) not in reliability, "gbm's zero must not be the comparison"
        assert "less than half as often" in reliability

    def test_the_reliability_line_is_omitted_when_it_would_not_be_true(self):
        rows = self.evaluation()
        rows[0] = pooled("news_markov", 0.253, n=255, n_tickers=5, fallback_rate=0.30)

        assert not [line for line in model_evaluation.trust_lines(rows) if "reliability" in line]

    def test_the_coverage_line_reports_the_measured_span(self):
        rows = self.evaluation()
        rows[1] = pooled("markov", 0.25335, n=255, n_tickers=5, coverage_80=0.839, fallback_rate=0.1686)

        line = next(l for l in model_evaluation.trust_lines(rows) if "coverage" in l)

        assert "0.82–0.84" in line
        assert "hit their 80% coverage target" in line

    def test_coverage_off_target_is_not_described_as_hitting_it(self):
        line = next(
            l for l in model_evaluation.trust_lines(self.evaluation(coverage_80=0.62)) if "coverage" in l
        )

        assert "hit" not in line
        assert "0.62" in line and "against an 80% coverage target" in line

    def test_a_fallback_forecast_says_so(self):
        lines = model_evaluation.trust_lines(self.evaluation(), forecast_row(model_used="gbm"))
        assert "Today's forecast used the fallback model." in lines

    def test_the_production_model_earns_no_notice(self):
        lines = model_evaluation.trust_lines(self.evaluation(), forecast_row())
        assert not [line for line in lines if "fallback model" in line]

    def test_an_unvalidated_forecast_says_it_is_unvalidated(self):
        lines = model_evaluation.trust_lines([], forecast_row())

        assert len(lines) == 1
        assert "has not been validated yet" in lines[0]

    def test_a_missing_baseline_does_not_claim_a_tie_against_it(self):
        """verdict() calls a missing arm indistinguishable; that is not a tie anyone measured."""
        rows = [pooled("news_markov", 0.253, n=255, n_tickers=5)]

        assert "no simpler baseline scored" in model_evaluation.trust_lines(rows)[0]

    def test_it_points_at_the_readme_for_the_names_it_withholds(self):
        assert model_evaluation.README_URL.startswith("https://github.com/")


class TestSuggestedQuestions:
    def test_there_are_three_and_they_ask_about_this_system(self):
        assert len(research_agent.SUGGESTED_QUESTIONS) == 3
        assert any("news" in question.lower() for question in research_agent.SUGGESTED_QUESTIONS)
        assert any(
            "can't" in question.lower() or "cannot" in question.lower()
            for question in research_agent.SUGGESTED_QUESTIONS
        )

    def test_a_click_queues_the_question_and_taking_it_consumes_it(self, monkeypatch):
        """Popped, not read: a Streamlit rerun must not re-ask the same thing."""
        monkeypatch.setattr(research_agent.st, "session_state", {})

        research_agent.queue_question("Why is this forecast positive?")
        assert research_agent.take_pending_question() == "Why is this forecast positive?"
        assert research_agent.take_pending_question() is None

    def test_resetting_the_conversation_drops_a_queued_question(self, monkeypatch):
        monkeypatch.setattr(research_agent.st, "session_state", {})

        research_agent.queue_question("What recent news is driving sentiment?")
        research_agent.reset_conversation()

        assert research_agent.take_pending_question() is None


class TestUiCopyGivesNoAdvice:
    """Spec A2 as a test: this product explains, and the words it explains with are checked."""

    def test_the_suggested_questions_and_the_caption_are_clean(self):
        for question in research_agent.SUGGESTED_QUESTIONS:
            assert_no_advice_words(question)
        assert_no_advice_words(research_agent.AGENT_CAPTION)
        assert "does not give investment advice" in research_agent.AGENT_CAPTION

    @pytest.mark.parametrize("probability", [0.10, 0.50, 0.90])
    def test_every_headline_verdict_band_is_clean(self, probability):
        for regime in ({"prob_low_vol": 0.9, "prob_high_vol": 0.1}, {"prob_low_vol": 0.1, "prob_high_vol": 0.9}, None):
            assert_no_advice_words(
                market_research.headline_verdict(forecast_row(prob_positive=probability), regime)
            )

    def test_the_range_card_copy_is_clean(self):
        assert_no_advice_words(market_research.loss_line(forecast_row()))
        assert_no_advice_words(market_research.distribution_caption(forecast_row()))
        for row in market_research.range_rows(forecast_row()):
            assert_no_advice_words(row["label"])

    def test_the_trust_lines_are_clean(self):
        rows = TestTrustStory.evaluation()
        for line in model_evaluation.trust_lines(rows, forecast_row(model_used="gbm")):
            assert_no_advice_words(line)

    def test_the_word_check_would_actually_catch_something(self):
        """A banned-word test that cannot fail is decoration."""
        with pytest.raises(AssertionError):
            assert_no_advice_words("This looks like a good entry.")
        with pytest.raises(AssertionError):
            assert_no_advice_words("Consider buying.")
        # And does not fire on words that merely contain one.
        assert_no_advice_words("accuracy held up against the threshold")


@pytest.fixture(scope="module")
def manifest() -> dict:
    """app.yaml at the REPOSITORY ROOT, which is where a Databricks App looks for it.

    It sat in app/ until the source root question was settled: the app imports src/ and reads
    config/config.yaml, so the deployed tree has to be the whole repository, and the manifest has
    to be at the root of it. The root requirements.txt is the app's for the same reason.
    """
    return yaml.safe_load((REPO_ROOT / "app.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def env(manifest) -> dict:
    return {entry["name"]: entry["value"] for entry in manifest["env"]}


class TestAppManifest:
    """app.yaml is where "the app can't read Delta" is fixed, so its contents are asserted."""

    def test_it_starts_the_streamlit_entry_point(self, manifest):
        assert manifest["command"] == ["streamlit", "run", "app/app.py"]

    @pytest.mark.parametrize(
        "name",
        [
            "DATABRICKS_SERVER_HOSTNAME",
            "DATABRICKS_HTTP_PATH",
            "DATABRICKS_WAREHOUSE_ID",
            "CATALOG",
            "LAKEBASE_HOST",
            "LAKEBASE_USER",
            "LAKEBASE_DATABASE",
            "LAKEBASE_ENDPOINT",
            "TELEMETRY_MODE",
        ],
    )
    def test_the_required_variables_are_declared(self, env, name):
        assert name in env, f"app.yaml must declare {name}"

    def test_telemetry_is_log_because_an_app_has_no_spark(self, env):
        assert env["TELEMETRY_MODE"] == "log"

    def test_the_warehouse_settings_are_readable_by_the_delta_module(self, env):
        """The manifest is parsed by the same code the app uses, not by a second reader here."""
        from src.database import delta

        settings = delta.settings_from_env(env)

        assert settings.server_hostname == env["DATABRICKS_SERVER_HOSTNAME"]
        assert settings.http_path == env["DATABRICKS_HTTP_PATH"]
        assert settings.access_token is None  # no token in a config file, ever

    def test_the_lakebase_settings_are_readable_by_the_lakebase_module(self, env):
        from src.database import lakebase

        settings = lakebase.settings_from_env(env)

        assert settings.host == env["LAKEBASE_HOST"]
        assert settings.endpoint == env["LAKEBASE_ENDPOINT"]

    def test_there_is_no_password_or_token_anywhere_in_it(self, env):
        """Rule 5. Both connections mint short-lived credentials; neither stores one."""
        forbidden = ("PASSWORD", "SECRET", "TOKEN", "PGPASSWORD")
        assert not [name for name in env if any(word in name.upper() for word in forbidden)]

    def test_the_command_is_relative_to_the_repository_root(self, manifest):
        """Which is the app's source root, and therefore where this path is resolved from."""
        entry = manifest["command"][-1]

        assert entry == "app/app.py"
        assert (REPO_ROOT / entry).exists(), f"{entry} does not exist relative to the source root"

    def test_the_app_requirements_exclude_the_modelling_stack(self):
        """The app reads precomputed Gold; statsmodels in the container is a cold start for nothing."""
        text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        installs = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not [line for line in installs if line.startswith(("statsmodels", "exchange_calendars"))]
        assert any(line.startswith("streamlit") for line in installs)
        assert any(line.startswith("psycopg") for line in installs)


@pytest.fixture
def agent_result():
    """An ``AgentResult`` from a turn that checked the forecast and then searched news."""
    from src.agent.agent import AgentResult, ToolInvocation

    return AgentResult(
        text="Downside risk is elevated because the model is 73% in the turbulent regime.",
        tool_calls=[
            ToolInvocation(
                name="get_market_forecast",
                arguments={"ticker": "NVDA"},
                result={"found": True, "forecast_id": "forecast-123"},
                ok=True,
            ),
            ToolInvocation(
                name="search_market_news",
                arguments={"ticker": "NVDA", "query": "downside risk"},
                result={"results": []},
                ok=True,
            ),
        ],
    )
