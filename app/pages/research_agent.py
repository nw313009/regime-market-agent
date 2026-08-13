"""Research Agent page (spec A2, C-5).

A chat scoped to one ticker, over ``src/agent/agent.run_agent``. The agent reads Gold and the news
index through its four tools and explains what it found; it computes nothing and recommends
nothing (the system prompt in ``src/agent/prompts.py`` is where those rules are enforced).

WHAT THIS PAGE ADDS TO THE LOOP:

- The ticker scope. It is prepended to the question rather than left to the model to infer, so
  "why is downside risk elevated?" reaches the tools with a symbol attached.
- The tool activity trail. Every turn shows which tools ran — "checked the forecast", "searched
  news" — because an explanation whose provenance is invisible is indistinguishable from a
  fabricated one. :func:`tool_activity` is a pure function and is tested directly.
- The watchlist sidebar, which is the CDC demo's first move (spec A3): add AMD here, watch the row
  appear in Postgres, then in Delta after the next ``sync_lakebase_history``.
- A save-report button wired to ``save_research_report``, which stamps the report with the
  ``forecast_id`` the turn actually used, when the agent looked one up.

WRITES GO THROUGH THE TOOLS, NOT AROUND THEM. The sidebar buttons call ``update_watchlist`` and
``save_research_report`` — the same functions the model calls — so there is exactly one code path
into Lakebase and one place where a ticker is validated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402 — the path has to be set before the app imports

from app.common import LAKEBASE_HINT, config, seed_tickers  # noqa: E402
from src.agent import tools as agent_tools  # noqa: E402
from src.agent.agent import AgentResult, run_agent  # noqa: E402

TITLE = "Research Agent"

#: Tool name -> what to say it did. Past tense and plain language: this trail is read by whoever is
#: watching the demo, not by a developer reading a log.
TOOL_LABELS = {
    "get_market_forecast": "checked the forecast",
    "search_market_news": "searched news",
    "update_watchlist": "updated the watchlist",
    "save_research_report": "saved a report",
}

#: Session-state keys, named once so a typo cannot silently create a second conversation.
HISTORY_KEY = "agent_history"
TRANSCRIPT_KEY = "agent_transcript"
LAST_RESULT_KEY = "agent_last_result"


# ------------------------------------------------------------------ pure functions


def scoped_question(ticker: str, question: str) -> str:
    """Attach the selected ticker to the question the user actually typed.

    The scope is stated rather than implied: without it "why did risk jump?" is a question the
    model has to guess the subject of, and a wrong guess costs a tool call against the wrong
    symbol.
    """
    return f"[Ticker: {ticker}] {question.strip()}"


def tool_activity(result: AgentResult | None) -> list[str]:
    """The trail of what the agent did this turn, in order, with failures marked.

    Repeated calls are kept rather than collapsed: two news searches in one turn is a fact about
    how the answer was assembled.
    """
    if result is None:
        return []

    trail: list[str] = []
    for call in result.tool_calls:
        label = TOOL_LABELS.get(call.name, call.name)
        # The query when there is one, since "searched news (NVDA)" on a page already scoped to
        # NVDA says nothing; the ticker otherwise.
        detail = call.arguments.get("query") or call.arguments.get("ticker")
        entry = f"{label} ({detail})" if detail else label
        trail.append(entry if call.ok else f"{entry} — failed")
    if result.hit_iteration_limit:
        trail.append("stopped at the tool-call limit")
    return trail


def used_forecast_id(result: AgentResult | None) -> str | None:
    """The ``forecast_id`` the turn actually read, for the saved report to point at.

    Taken from the tool RESULT rather than from a separate query, so a saved report references the
    exact forecast the text describes even if the daily job wrote a new one in the meantime.
    """
    if result is None:
        return None
    for call in reversed(result.tool_calls):
        if call.name != "get_market_forecast" or not call.ok:
            continue
        forecast_id = call.result.get("forecast_id")
        if forecast_id:
            return str(forecast_id)
    return None


def transcript_entry(role: str, content: str, activity: Sequence[str] = ()) -> dict:
    return {"role": role, "content": content, "activity": list(activity)}


# ------------------------------------------------------------------ state


def state() -> tuple[list[dict], list[dict]]:
    """The model-facing message history and the human-facing transcript.

    Two lists, deliberately. The history holds tool calls and tool results the model needs and a
    reader does not; the transcript holds what belongs on screen. Rendering the raw history would
    put JSON tool payloads in the chat.
    """
    st.session_state.setdefault(HISTORY_KEY, [])
    st.session_state.setdefault(TRANSCRIPT_KEY, [])
    return st.session_state[HISTORY_KEY], st.session_state[TRANSCRIPT_KEY]


def reset_conversation() -> None:
    for key in (HISTORY_KEY, TRANSCRIPT_KEY, LAST_RESULT_KEY):
        st.session_state.pop(key, None)


def ask(ticker: str, question: str) -> AgentResult:
    """One turn: run the loop, keep the history it returns, and store the result for the sidebar."""
    history, transcript = state()
    result = run_agent(scoped_question(ticker, question), history, config=config())

    st.session_state[HISTORY_KEY] = result.messages
    transcript.append(transcript_entry("user", question))
    transcript.append(transcript_entry("assistant", result.text, tool_activity(result)))
    st.session_state[LAST_RESULT_KEY] = result
    return result


# ------------------------------------------------------------------ rendering


def render() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":speech_balloon:", layout="wide")
    st.title(TITLE)

    ticker = _render_sidebar()
    st.caption(
        f"Scoped to **{ticker}**. The agent reads `gold.forecast_runs`, `gold.regime_states` and "
        "the news index. It does not compute statistics and does not give advice."
    )

    _render_transcript()

    question = st.chat_input(f"Ask about {ticker}")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Reading the forecast and the news..."):
            try:
                result = ask(ticker, question)
            except Exception as exc:  # noqa: BLE001 — a failed turn must not clear the chat
                st.error(f"The agent could not finish this turn: {type(exc).__name__}: {exc}")
                return
            _render_activity(tool_activity(result))
            st.markdown(result.text)


def _render_transcript() -> None:
    _, transcript = state()
    for entry in transcript:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                _render_activity(entry.get("activity", ()))
            st.markdown(entry["content"])


def _render_activity(activity: Sequence[str]) -> None:
    if activity:
        st.caption("· ".join(activity))


def _render_sidebar() -> str:
    """Ticker scope, watchlist add/remove, save-report. The whole write surface of the app."""
    with st.sidebar:
        st.header("Watchlist")
        watchlist = _watchlist()
        options = sorted(set(seed_tickers()) | set(watchlist))
        ticker = st.selectbox("Ticker", options, index=0, key="agent_ticker")

        st.write(", ".join(watchlist) if watchlist else "_empty_")

        candidate = st.text_input("Symbol", key="watchlist_symbol", placeholder="AMD").strip()
        add, remove = st.columns(2)
        if add.button("Add", use_container_width=True, disabled=not candidate):
            _watchlist_write("add", candidate)
        if remove.button("Remove", use_container_width=True, disabled=not candidate):
            _watchlist_write("remove", candidate)

        st.divider()
        st.header("Report")
        result: AgentResult | None = st.session_state.get(LAST_RESULT_KEY)
        _, transcript = state()
        last_question = next(
            (entry["content"] for entry in reversed(transcript) if entry["role"] == "user"), ""
        )
        if st.button("Save the last answer", use_container_width=True, disabled=result is None):
            _save_report(ticker, last_question, result)

        st.divider()
        if st.button("New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        return ticker


def _watchlist() -> list[str]:
    try:
        from src.database import lakebase

        return lakebase.get_watchlist()
    except Exception as exc:  # noqa: BLE001 — degrade to seed tickers, never blank the page
        st.caption(f"{LAKEBASE_HINT} ({type(exc).__name__}: {exc})")
        return []


def _watchlist_write(action: str, symbol: str) -> None:
    """Add or remove through the TOOL, so the button and the model share one code path."""
    try:
        result = agent_tools.update_watchlist(action, symbol)
    except agent_tools.ToolError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"{LAKEBASE_HINT} ({type(exc).__name__}: {exc})")
        return

    # The tool's own message, not a sentence built here: it reports whether anything actually
    # CHANGED, and "AMD was already on the watchlist" is the honest answer to a second click.
    st.success(f"{result['message']} Now: {', '.join(result['watchlist']) or 'empty'}")
    st.rerun()


def _save_report(ticker: str, question: str, result: AgentResult | None) -> None:
    if result is None or not result.text.strip():
        st.warning("Ask something first — there is no answer to save.")
        return
    try:
        saved = agent_tools.save_research_report(
            ticker,
            question,
            result.text,
            forecast_id=used_forecast_id(result),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"{LAKEBASE_HINT} ({type(exc).__name__}: {exc})")
        return
    st.success(f"Saved report {saved['report_id']}")


if __name__ == "__main__":
    render()
