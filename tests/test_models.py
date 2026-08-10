"""Markov model tests (spec B-7). All mandatory before Checkpoint B freezes.

- ``test_transition_orientation``: for any fitted P, assert ``np.allclose(P.sum(axis=0), 1)``.
  The matrix is left-stochastic, so COLUMNS sum to 1.
- ``test_stationary``: solve for pi from P as the left eigenvector under the column
  convention, then simulate 200k steps with the production sampler and assert the empirical
  frequencies land within 1% of pi. This is the test that catches a transposed sampler
  instantly, and it is the reason it exists.
- ``test_fallback``: inject a ``FitError`` from Model C and assert Model B was used AND that
  the substitution was recorded. A silent fallback is as bad as a crash.
- Regime sorting: assert the low-volatility regime is always index 0 after
  ``sort_regimes``, and that ``mus``, ``sigmas``, ``P`` and the filtered probabilities were
  all permuted consistently.
- Degeneracy: assert the sigma-ratio and diagonal-probability checks raise ``FitError`` so
  the ladder actually descends.

TODO: implement.
"""
