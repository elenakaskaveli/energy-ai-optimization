"""Fetch historical hourly weather data from the Open-Meteo Archive API."""

import logging

import pandas as pd
import requests

from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_FILENAME = "weather.csv"


def _fetch_year(config: dict, start: str, end: str) -> pd.DataFrame:
    """One archive-API call for a single year's date range (the API doesn't
    accept the full multi-year span in one request)."""
    params = {
        "latitude": config["location"]["latitude"],
        "longitude": config["location"]["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(config["weather"]["hourly_variables"]),
        "timezone": config["location"]["timezone"],
    }
    response = requests.get(config["weather"]["archive_endpoint"], params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(hourly)


def fetch_weather(config: dict | None = None) -> None:
    """Download historical weather one year at a time and concatenate into one CSV."""
    config = config or load_config()
    raw_dir = resolve_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / RAW_FILENAME

    if target_path.exists():
        logger.info("Weather data already present at %s, skipping download.", target_path)
        return

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    years = pd.period_range(start=start_date, end=end_date, freq="Y")

    frames = []
    for year in years:
        year_start = max(pd.Timestamp(start_date), year.start_time).date().isoformat()
        year_end = min(pd.Timestamp(end_date), year.end_time).date().isoformat()
        logger.info("Fetching weather for %s .. %s", year_start, year_end)
        frames.append(_fetch_year(config, year_start, year_end))

    weather = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    weather.to_csv(target_path, index=False)
    logger.info("Saved weather data (%d rows) to %s", len(weather), target_path)


if __name__ == "__main__":
    fetch_weather()
