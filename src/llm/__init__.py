"""Model access and telemetry (spec C-3).

There is no routing subsystem. Every model call goes through one lightweight abstraction,
``call_model(task, ...)``, and configuration decides which Databricks endpoint that means.

Explicitly not here: model tiers, semantic routing, AI Gateway, escalation graphs, routing
benchmarks, middleware.
"""
