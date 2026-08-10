# Databricks notebook source
"""Walk-forward backtest runner (spec B-5).

Run on demand, NOT part of the daily job. Results land in ``gold.backtest_metrics`` and the
Model Evaluation page reads them from there.

A thin wrapper only: %pip install -r requirements.txt (statsmodels and exchange_calendars are
the ones that are not preinstalled), append the repo root to sys.path, then call
``src.models.backtest``. No logic in this file.
"""
