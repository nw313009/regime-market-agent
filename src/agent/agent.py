"""The agent loop (spec C-4).

A plain tool-calling loop, capped at roughly 6 tool iterations:

1. Send the system prompt, the conversation history and the tool schemas via
   ``call_model("agent", ...)``.
2. Execute whatever tool calls come back.
3. Append the tool results to the history.
4. Stop when the model returns a final text answer, or when the iteration cap is hit.

A typical turn for "why is downside risk elevated this week?" is
``get_market_forecast`` -> ``search_market_news`` -> a written explanation that cites both
the Gold numbers and the retrieved article titles, and invents neither.

No routing, no escalation, no sub-agents.
"""
