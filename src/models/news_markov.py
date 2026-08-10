"""Model C — News-Markov with time-varying transition probabilities (spec B-3).

Model C is Model B plus news sentiment as an exogenous driver of the transition
probabilities. It answers one question, without presuming the answer is yes:

    Does recent financial news improve five-trading-day probabilistic forecasting beyond
    price-driven regime switching alone?

No LLM-generated feature ever enters this path.

exog_tvtp construction — THE alignment rule::

    n = df["news_sentiment_3d"].shift(1)          # lag one trading day
    exog_tvtp = np.column_stack([np.ones(len(n)), n])
    # drop the first row jointly with endog so the lengths match

statsmodels uses ``exog_tvtp`` row t to build the transition INTO t, i.e. the transition
from state t-1 to state t. Shifting by one day is what makes "news known at t affects the
t -> t+1 transition" actually true, instead of letting the model peek at same-day news.

The column of ones is mandatory: it is the intercept of the transition model.

A "lengths differ" error on ``exog_tvtp`` means the ``shift(1)`` row was not dropped
jointly with ``endog``.

The no-lookahead test in ``tests/test_no_lookahead.py`` is what proves the alignment.
"""
