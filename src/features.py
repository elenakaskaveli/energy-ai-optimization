"""Feature engineering for demand forecasting: calendar, lag, and rolling features."""

import holidays
import pandas as pd


def add_calendar_features(df: pd.DataFrame, country: str = "FR") -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    country_holidays = holidays.country_holidays(country)
    df["is_holiday"] = [int(date in country_holidays) for date in idx.date]
    return df


def add_lag_features(df: pd.DataFrame, target_column: str, lags=(1, 24, 168)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_column}_lag_{lag}h"] = df[target_column].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_column: str, windows=(24, 168), min_lag: int = 1
) -> pd.DataFrame:
    """Rolling mean/std of `target_column` over each window, computed from data at
    least `min_lag` hours before the current row (no peeking at more recent history
    than a forecast would actually have available)."""
    df = df.copy()
    shifted = df[target_column].shift(min_lag)
    for window in windows:
        df[f"{target_column}_rollmean_{window}h"] = shifted.rolling(window).mean()
        df[f"{target_column}_rollstd_{window}h"] = shifted.rolling(window).std()
    return df


def build_features(
    df: pd.DataFrame,
    target_column: str,
    weather_columns=("temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"),
    lags=(1, 24, 168),
    rolling_windows=(24, 168),
    country: str = "FR",
) -> pd.DataFrame:
    """Build a supervised-learning feature table from the raw hourly dataset.

    `lags` sets both which lag columns are created and, via `min(lags)`, how far
    before the target the rolling-window features are allowed to look — pass e.g.
    `lags=(24, 48, 168)` (no 1-hour lag) to simulate genuine day-ahead forecasting,
    where the previous hour's actual value isn't known yet.

    Rows with NaN features (from lag/rolling warm-up) are dropped.
    """
    feature_df = add_calendar_features(df[[target_column, *weather_columns]], country=country)
    feature_df = add_lag_features(feature_df, target_column, lags=lags)
    feature_df = add_rolling_features(feature_df, target_column, windows=rolling_windows, min_lag=min(lags))
    return feature_df.dropna()


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    return [c for c in df.columns if c != target_column]
