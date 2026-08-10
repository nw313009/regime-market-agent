"""The single model-access abstraction (spec C-3).

Contract::

    def call_model(task: str, messages, tools=None, response_format=None) -> Response

Reads the endpoint name from config by task: ``"agent"`` -> ``model.agent_endpoint`` now,
``"slm"`` -> ``model.slm_endpoint`` later if the stretch goal happens. Wraps the Databricks
Foundation Model API, which is OpenAI-compatible chat completions.

No routing logic. No tiers. Endpoint names must not be scattered as literals through the
repository — that is the entire reason this indirection exists.
"""
