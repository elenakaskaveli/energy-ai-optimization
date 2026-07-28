"""Seasonal naive baseline: predict the value observed one day (or one week) earlier."""

import numpy as np
import pandas as pd


def seasonal_naive_predict(df: pd.DataFrame, lag_column: str) -> np.ndarray:
    """Use an already-computed lag feature (e.g. `<target>_lag_24h`) as the forecast."""
    return df[lag_column].to_numpy()
