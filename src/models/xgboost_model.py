"""XGBoost regressor for next-hour demand forecasting from lag/calendar/weather features."""

import pandas as pd
from xgboost import XGBRegressor


class XGBoostForecaster:
    """Gradient-boosted decision trees: each new tree is fit to correct the
    residual error of all previous trees combined, so the final prediction is
    the sum of every tree's contribution.

    Default hyperparameters below are reasonable starting values; `run_forecasting.py`
    overrides them with `_tune_xgboost`'s grid-search result for the model that ships.
    """

    def __init__(self, **kwargs):
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = XGBRegressor(**params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostForecaster":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame):
        return self.model.predict(X_test)

    def feature_importances(self, feature_names) -> pd.Series:
        """Per-feature importance scores, most influential first."""
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
