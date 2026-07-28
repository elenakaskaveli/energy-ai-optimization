"""Merge household consumption and weather data into an hourly modeling dataset."""

import logging

import pandas as pd

from src.config import load_config, resolve_path
from src.data.fetch_consumption import RAW_FILENAME as CONSUMPTION_FILENAME
from src.data.fetch_weather import RAW_FILENAME as WEATHER_FILENAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_FILENAME = "household_energy_hourly.csv"

CONSUMPTION_COLUMNS = {
    "Global_active_power": "global_active_power_kw",
    "Global_reactive_power": "global_reactive_power_kw",
    "Voltage": "voltage_v",
    "Global_intensity": "global_intensity_a",
    "Sub_metering_1": "sub_metering_1_wh",
    "Sub_metering_2": "sub_metering_2_wh",
    "Sub_metering_3": "sub_metering_3_wh",
}


def _load_consumption(raw_dir, resample_freq: str) -> pd.DataFrame:
    path = raw_dir / CONSUMPTION_FILENAME
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.set_index("timestamp").drop(columns=["Date", "Time"])
    df = df.rename(columns=CONSUMPTION_COLUMNS)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.resample(resample_freq).mean()


def _load_weather(raw_dir) -> pd.DataFrame:
    path = raw_dir / WEATHER_FILENAME
    df = pd.read_csv(path, parse_dates=["time"])
    return df.set_index("time").rename_axis("timestamp")


def build_dataset(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    raw_dir = resolve_path(config["data"]["raw_dir"])
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    consumption = _load_consumption(raw_dir, config["data"]["resample_freq"])
    weather = _load_weather(raw_dir)

    merged = consumption.join(weather, how="inner")
    merged = merged.dropna(subset=[config["forecasting"]["target_column"]])

    target_path = processed_dir / PROCESSED_FILENAME
    merged.to_csv(target_path)
    logger.info("Saved merged hourly dataset (%d rows, %d cols) to %s", *merged.shape, target_path)
    return merged


if __name__ == "__main__":
    build_dataset()
