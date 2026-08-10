"""Lightweight model-call telemetry (spec C-3).

Appends one record per model call to the Delta table ``gold.model_calls``::

    {ts, task, model, latency_ms, ok, in_tokens, out_tokens}

Token counts are recorded when the endpoint reports them.

This is instrumentation, not a separate observability project. No AI-routing dashboard.
"""
