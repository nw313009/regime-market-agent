"""The research agent: exactly one agent, exactly four tools (spec C-4).

Architectural division of labour::

    Python / statistical system  =  numerical inference
    LLM                          =  reasoning + orchestration + explanation

The agent retrieves, synthesizes, explains and persists user actions. It never computes a
Markov model or a Monte Carlo forecast — it reads Gold and explains it.

No multi-agent orchestration, no router, no escalation graph.
"""
