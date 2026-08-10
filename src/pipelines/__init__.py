"""Spark transformations: bronze -> silver -> ``silver.daily_features``.

``silver.daily_features`` is the contract between Spark and the modeling layer. Feature
scope stays deliberately small; do not expand it while the models are unvalidated.
"""
