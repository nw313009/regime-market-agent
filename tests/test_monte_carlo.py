"""Monte Carlo tests (spec B-7).

- ``test_mc_seed``: the same seed must produce identical percentiles; a different seed must
  produce different ones. Both halves matter — the second catches a Generator that is not
  actually being used.
- Assert the sampler reads ``P_h[:, current_regime]``, i.e. the previous state selects a
  COLUMN. Pair this with ``test_stationary`` in ``test_models.py``.
- Assert the news decay follows ``news_h = current_news * exp(-ln(2)/half_life * h)``.
- Assert the output carries ``n_paths``, ``seed`` and ``model_version``, and that raw paths
  are NOT persisted.
- Assert returns are simulated on the decimal scale: fitted percent-scale mu and sigma must
  have been divided by 100 before simulation.

TODO: implement.
"""
