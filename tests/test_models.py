import numpy as np
import pandas as pd

from src.models.baseline import seasonal_naive_predict
from src.models.lstm_model import LSTMForecaster, create_sequences
from src.models.statistical import HoltWintersForecaster
from src.models.xgboost_model import XGBoostForecaster


def test_seasonal_naive_predict_returns_lag_column():
    df = pd.DataFrame({"target_lag_24h": [1.0, 2.0, 3.0]})
    np.testing.assert_array_equal(seasonal_naive_predict(df, "target_lag_24h"), [1.0, 2.0, 3.0])


def test_xgboost_forecaster_fit_predict_shapes():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y_train = X_train["a"] * 2 + 1
    model = XGBoostForecaster(n_estimators=10)
    model.fit(X_train, y_train)

    X_test = pd.DataFrame(rng.normal(size=(5, 3)), columns=["a", "b", "c"])
    preds = model.predict(X_test)
    assert len(preds) == 5

    importances = model.feature_importances(["a", "b", "c"])
    assert set(importances.index) == {"a", "b", "c"}


def test_holt_winters_forecaster_predict_length():
    idx = pd.date_range("2020-01-01", periods=72, freq="1h")
    series = pd.Series(np.sin(np.arange(72) / 24 * 2 * np.pi) + 5, index=idx)
    model = HoltWintersForecaster(seasonal_periods=24).fit(series)
    preds = model.predict(steps=12)
    assert len(preds) == 12


def test_create_sequences_shapes():
    features = np.arange(20).reshape(10, 2).astype(float)
    target = np.arange(10).astype(float)
    X_seq, y_seq = create_sequences(features, target, lookback=3)
    assert X_seq.shape == (7, 3, 2)
    assert y_seq.shape == (7,)


def test_lstm_forecaster_fit_predict_smoke():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 2))
    target = features[:, 0] * 2 + 1

    model = LSTMForecaster(lookback=5, units=4, epochs=1, batch_size=16)
    model.fit(features, target)

    preds = model.predict(features)
    assert len(preds) == len(features) - 5
