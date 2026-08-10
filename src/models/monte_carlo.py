"""Monte Carlo forecast simulation (spec B-4).

Contract::

    def run_forecast(sorted_params, model_res_or_none, current_price,
                     current_news, cfg, rng) -> ForecastSummary

5,000 paths x 5 trading days. Per path, for each day h in 1..5::

    news_h = current_news * exp(-ln(2) / half_life * h)

    if TVTP model:
        P_h = res.model.regime_transition_matrix(
            res.params, exog_tvtp=np.array([[1.0, news_h]]))[:, :, 0]
        # then apply the SAME perm that sort_regimes used
    else:
        P_h = the sorted static P

    next_regime ~ Categorical(P_h[:, current_regime])   # COLUMN = previous state
    r ~ Normal(mus[next_regime], sigmas[next_regime])   # decimal scale, not percent
    price *= exp(r)
    current_regime = next_regime

The column indexing is the whole ballgame: the matrix is left-stochastic, so the previous
state selects a COLUMN. A sampler that indexes rows is transposed, and the stationary test
in ``tests/test_models.py`` catches it immediately.

Initial regime per path ~ ``Categorical(filtered_current)``.

GBM paths: ``r ~ Normal(mu, sigma)``, no regimes.

Randomness: ``rng = np.random.default_rng(cfg.forecast.seed)`` — one Generator per forecast
run, and the seed is stored with the output so the run is reproducible.

Outputs: price and return P10/P50/P90; ``prob_positive`` = ``mean(R5 > 0)``;
``prob_loss_gt_5pct``; regime probabilities; ``n_paths``; ``model_version``. Emphasize the
distribution, not a point prediction.

Do NOT persist the raw 5,000 paths.

News decay is an assumption, not a measurement, so the UI discloses it in a sentence.
"""
