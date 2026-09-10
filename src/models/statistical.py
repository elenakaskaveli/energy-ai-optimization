"""Classical statistical baseline: Holt-Winters exponential smoothing with daily seasonality."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersForecaster:
    """Triple exponential smoothing: tracks level, trend, and a repeating seasonal
    pattern (here, the daily consumption cycle), each updated with exponentially
    decaying weight on older observations."""

    def __init__(self, seasonal_periods: int = 24):
        self.seasonal_periods = seasonal_periods
        self._fit_result = None

    def fit(self, train_series: pd.Series) -> "HoltWintersForecaster":
        """Fit level, trend, and seasonal components on a univariate training series."""
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
        """Forecast the next `steps` values beyond the end of the training series."""
        return self._fit_result.forecast(steps).to_numpy()
