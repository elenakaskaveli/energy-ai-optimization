import numpy as np
import pandas as pd

from src.features import add_calendar_features, add_lag_features, add_rolling_features, build_features


def _sample_df(n=200):
    idx = pd.date_range("2007-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "global_active_power_kw": np.sin(np.arange(n) / 24 * 2 * np.pi) + 3,
            "temperature_2m": np.random.default_rng(0).normal(10, 5, n),
            "shortwave_radiation": np.random.default_rng(1).uniform(0, 500, n),
            "wind_speed_10m": np.random.default_rng(2).uniform(0, 20, n),
            "cloud_cover": np.random.default_rng(3).integers(0, 100, n),
        },
        index=idx,
    )


def test_add_calendar_features_known_dates():
    df = _sample_df(48)
    out = add_calendar_features(df, country="FR")

    monday_2007_01_01 = pd.Timestamp("2007-01-01")
    assert monday_2007_01_01.dayofweek == 0
    assert out.loc["2007-01-01 00:00:00", "day_of_week"] == 0
    assert out.loc["2007-01-01 00:00:00", "hour"] == 0
    assert out.loc["2007-01-01 00:00:00", "is_weekend"] == 0
    # 2007-01-01 is New Year's Day in France
    assert out.loc["2007-01-01 00:00:00", "is_holiday"] == 1


def test_add_lag_features_shifts_correctly():
    df = _sample_df(10)
    out = add_lag_features(df, "global_active_power_kw", lags=(1,))
    shifted = out["global_active_power_kw_lag_1h"]
    assert pd.isna(shifted.iloc[0])
    assert shifted.iloc[1] == df["global_active_power_kw"].iloc[0]


def test_add_rolling_features_uses_shifted_window():
    df = _sample_df(30)
    out = add_rolling_features(df, "global_active_power_kw", windows=(5,))
    # rolling mean at row 5 should equal mean of rows [0..4] (shifted by 1, no leakage of row 5 itself)
    expected = df["global_active_power_kw"].iloc[0:5].mean()
    assert np.isclose(out["global_active_power_kw_rollmean_5h"].iloc[5], expected)


def test_build_features_drops_warmup_nans():
    df = _sample_df(300)
    features = build_features(df, target_column="global_active_power_kw", lags=(1, 24), rolling_windows=(24,))
    assert features.isna().sum().sum() == 0
    assert len(features) == len(df) - 24
