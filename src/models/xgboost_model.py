"""XGBoost regressor for next-hour demand forecasting from lag/calendar/weather features."""

import pandas as pd
from xgboost import XGBRegressor


class XGBoostForecaster:
    def __init__(self, **kwargs):
        params = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        params.update(kwargs)
        self.model = XGBRegressor(**params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostForecaster":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame):
        return self.model.predict(X_test)

    def feature_importances(self, feature_names) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
