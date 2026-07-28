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


def evaluate(y_true, y_pred) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """results: {model_name: {"RMSE": ..., "MAE": ..., "MAPE": ...}}"""
    return pd.DataFrame(results).T.sort_values("RMSE")
