# Data sources

This project does not commit raw data to the repository. Run the scripts in
`src/data/` to fetch and build the datasets locally.

## 1. Household electricity consumption

- Source: [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- Coverage: a single household in Sceaux, France, December 2006 – November 2010, minute-level resolution.
- Fetched and cached to `data/raw/household_power_consumption.txt` by `src/data/fetch_consumption.py`.

## 2. Weather data

- Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (free, no API key required).
- Location: Sceaux, France (48.778, 2.290), matching the consumption dataset.
- Variables: temperature, shortwave radiation, wind speed, cloud cover.
- Fetched and cached to `data/raw/weather.csv` by `src/data/fetch_weather.py`.

## 3. Processed dataset

- `src/data/build_dataset.py` resamples consumption to hourly resolution, joins it with weather
  on timestamp, and writes the merged dataset to `data/processed/household_energy_hourly.csv`.

All parameters (location, date ranges, resample frequency) are controlled from
[`config/config.yaml`](../config/config.yaml).
