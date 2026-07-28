"""LSTM sequence model for next-hour demand forecasting."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback : i])
        y.append(target[i])
    return np.array(X), np.array(y)


class LSTMForecaster:
    """Predicts the next value in `target` from the previous `lookback` hours of `features`.

    `predict` returns one fewer than `len(features)` values if `features` was not
    prefixed with `lookback` rows of prior context: the first `lookback` rows of any
    input array have no history and are dropped.
    """

    def __init__(self, lookback: int = 24, units: int = 32, epochs: int = 10, batch_size: int = 64, random_seed: int = 42):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None

    def _build_model(self, n_features: int):
        keras.utils.set_random_seed(self.random_seed)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.lookback, n_features)),
                keras.layers.LSTM(self.units),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LSTMForecaster":
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).ravel()
        X_seq, y_seq = create_sequences(features_scaled, target_scaled, self.lookback)
        self.model = self._build_model(features.shape[1])
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features_scaled = self.feature_scaler.transform(features)
        dummy_target = np.zeros(len(features_scaled))
        X_seq, _ = create_sequences(features_scaled, dummy_target, self.lookback)
        preds_scaled = self.model.predict(X_seq, verbose=0).ravel()
        return self.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
