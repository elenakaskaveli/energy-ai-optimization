"""Classical statistical baseline: Holt-Winters exponential smoothing with daily seasonality."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersForecaster:
    def __init__(self, seasonal_periods: int = 24):
        self.seasonal_periods = seasonal_periods
        self._fit_result = None

    def fit(self, train_series: pd.Series) -> "HoltWintersForecaster":
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._fit_result = model.fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self._fit_result.forecast(steps).to_numpy()
