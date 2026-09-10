"""Forecast accuracy metrics and model comparison utilities."""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred, epsilon: float = 1e-3) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), epsilon, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted (aggregate) absolute percentage error: sum of errors over sum of
    actuals, rather than averaging per-hour ratios. Unlike MAPE, a handful of
    near-zero actual readings (e.g. household demand at 3am) can't blow up the
    result, since each hour is weighted by its own magnitude instead of
    contributing an unbounded ratio."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """results: {model_name: {"RMSE": ..., "MAE": ..., "MAPE": ...}}"""
    return pd.DataFrame(results).T.sort_values("RMSE")
