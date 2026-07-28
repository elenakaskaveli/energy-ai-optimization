"""Train and compare demand forecasting models: seasonal naive, Holt-Winters, XGBoost, LSTM.

XGBoost is evaluated under two scenarios to make the forecasting task's
difficulty explicit: a "nowcast" that may use the actual value from 1 hour ago
(a live smart-meter reading), and a "day-ahead" scenario that only has access
to data from 24+ hours before the target — the realistic case for scheduling
battery charge/discharge a day in advance.
"""

import json
import logging

# Must be imported before xgboost/shap: on this platform, letting xgboost's
# bundled OpenMP runtime initialize first causes tensorflow's threading setup
# to deadlock the first time a Keras model is trained later in the process.
import tensorflow  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.config import load_config, resolve_path
from src.evaluation.metrics import compare_models, evaluate
from src.features import build_features, get_feature_columns
from src.models.baseline import seasonal_naive_predict
from src.models.lstm_model import LSTMForecaster
from src.models.statistical import HoltWintersForecaster
from src.models.xgboost_model import XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LSTM_FEATURE_COLUMNS = [
    "temperature_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "cloud_cover",
    "hour",
    "day_of_week",
]


def load_processed_dataset(config: dict) -> pd.DataFrame:
    path = resolve_path(config["data"]["processed_dir"]) / "household_energy_hourly.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def main():
    config = load_config()
    target_column = config["forecasting"]["target_column"]
    split_date = config["data"]["train_test_split_date"]

    raw = load_processed_dataset(config)

    # Two forecasting scenarios, differing only in which lag/rolling features are
    # allowed:
    #  - nowcast: includes the actual value from 1 hour ago (realistic for a live
    #    smart-meter feed, but NOT usable to plan a full day ahead).
    #  - day-ahead: excludes anything more recent than 24h before the target, i.e.
    #    the earliest lag is "same hour yesterday" — this is the realistic input
    #    available when scheduling battery charge/discharge a day in advance.
    nowcast_features = build_features(raw, target_column=target_column, lags=(1, 24, 168))
    day_ahead_features = build_features(raw, target_column=target_column, lags=(24, 48, 168))

    def split(features_df):
        train = features_df[features_df.index < split_date]
        test = features_df[features_df.index >= split_date]
        return train, test

    nowcast_train, nowcast_test = split(nowcast_features)
    day_ahead_train, day_ahead_test = split(day_ahead_features)
    logger.info("Train: %d rows, Test: %d rows", len(nowcast_train), len(nowcast_test))

    results = {}
    predictions = {}

    # 1. Seasonal naive baseline (yesterday, same hour) — legitimate for day-ahead too
    naive_preds = seasonal_naive_predict(day_ahead_test, lag_column=f"{target_column}_lag_24h")
    results["Seasonal Naive (day-ahead)"] = evaluate(day_ahead_test[target_column], naive_preds)
    predictions["Seasonal Naive"] = naive_preds

    # 2. Holt-Winters exponential smoothing (classical statistical baseline)
    hw = HoltWintersForecaster(seasonal_periods=24).fit(nowcast_train[target_column])
    hw_preds = hw.predict(steps=len(nowcast_test))
    results["Holt-Winters"] = evaluate(nowcast_test[target_column], hw_preds)
    predictions["Holt-Winters"] = hw_preds

    # 3a. XGBoost — nowcast (uses last hour's real reading; not valid day-ahead)
    nowcast_feature_columns = get_feature_columns(nowcast_features, target_column)
    xgb_nowcast = XGBoostForecaster().fit(
        nowcast_train[nowcast_feature_columns], nowcast_train[target_column]
    )
    xgb_nowcast_preds = xgb_nowcast.predict(nowcast_test[nowcast_feature_columns])
    results["XGBoost (nowcast, uses last hour)"] = evaluate(nowcast_test[target_column], xgb_nowcast_preds)
    predictions["XGBoost (nowcast)"] = xgb_nowcast_preds

    # 3b. XGBoost — day-ahead (no access to anything within 24h of the target)
    day_ahead_feature_columns = get_feature_columns(day_ahead_features, target_column)
    xgb_day_ahead = XGBoostForecaster().fit(
        day_ahead_train[day_ahead_feature_columns], day_ahead_train[target_column]
    )
    xgb_day_ahead_preds = xgb_day_ahead.predict(day_ahead_test[day_ahead_feature_columns])
    results["XGBoost (day-ahead)"] = evaluate(day_ahead_test[target_column], xgb_day_ahead_preds)
    predictions["XGBoost (day-ahead)"] = xgb_day_ahead_preds

    top_features = xgb_day_ahead.feature_importances(day_ahead_feature_columns).head(10)
    logger.info("Top 10 XGBoost (day-ahead) feature importances:\n%s", top_features)

    _plot_shap_summary(
        xgb_day_ahead.model,
        day_ahead_test[day_ahead_feature_columns],
        resolve_path("reports/figures") / "shap_summary.png",
    )

    train, test = nowcast_train, nowcast_test
    y_test = test[target_column]

    # 4. LSTM
    lookback = 24
    lstm = LSTMForecaster(lookback=lookback, units=32, epochs=10)
    lstm.fit(train[LSTM_FEATURE_COLUMNS].to_numpy(), train[target_column].to_numpy())

    context = pd.concat([train.tail(lookback), test])
    lstm_preds = lstm.predict(context[LSTM_FEATURE_COLUMNS].to_numpy())
    # first `lookback` rows of `test` have no LSTM prediction context beyond train tail;
    # since we prefixed with train tail, predictions align 1:1 with all of `test`.
    results["LSTM"] = evaluate(y_test, lstm_preds)
    predictions["LSTM"] = lstm_preds

    comparison = compare_models(results)
    logger.info("Model comparison:\n%s", comparison)

    output_dir = resolve_path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(resolve_path("reports") / "forecasting_model_comparison.csv")

    predictions_df = pd.DataFrame({"actual": y_test, **predictions}, index=y_test.index)
    predictions_df.to_csv(resolve_path("reports") / "forecasting_predictions.csv")

    with open(resolve_path("reports") / "forecasting_top_features.json", "w") as f:
        json.dump(top_features.to_dict(), f, indent=2)

    _plot_predictions(y_test, predictions, output_dir / "forecast_comparison.png")

    return comparison


def _plot_shap_summary(xgb_model, X_test: pd.DataFrame, output_path, sample_size: int = 1000):
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", output_path)


def _plot_predictions(y_test: pd.Series, predictions: dict, output_path):
    sample = slice(0, 24 * 7)  # first week of the test set
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index[sample], y_test.iloc[sample], label="Actual", color="black", linewidth=1.5)
    for name, preds in predictions.items():
        ax.plot(y_test.index[sample], preds[sample], label=name, alpha=0.8)
    ax.set_title("Household demand forecast — first week of test period")
    ax.set_ylabel("Global active power (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    logger.info("Saved comparison plot to %s", output_path)


if __name__ == "__main__":
    main()
