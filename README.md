# Smart Home Energy Management System

AI-driven system for a residential household that forecasts electricity demand,
optimizes battery storage scheduling, and recommends solar PV integration —
built on real consumption and weather data.

> Status: in development. This README is updated as each component lands.

## Overview

Three components work together on a single-household scenario:

1. **Demand forecasting** — predicts hourly household electricity consumption
   using calendar and weather features. Baseline (seasonal naive / SARIMA) is
   compared against XGBoost and an LSTM sequence model.
2. **Battery storage optimization** — given the demand forecast and estimated
   solar generation, a linear program schedules battery charge/discharge to
   minimize electricity cost and maximize self-consumption of solar energy.
3. **Renewable integration analysis** — estimates rooftop solar PV output from
   historical irradiance and recommends system sizing to reach target
   self-sufficiency levels.

## Data

- **Consumption**: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (Sceaux, France, 2006–2010).
- **Weather**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) for the same location.
- **Solar PV**: modeled with [`pvlib`](https://pvlib-python.readthedocs.io/) from historical irradiance.

See [`data/README.md`](data/README.md) for details and fetch scripts.

## Project structure

```
energy-ai-optimization/
├── config/config.yaml      # all tunable parameters (location, battery, tariff, models)
├── data/                   # raw/ and processed/ datasets (not committed, fetched locally)
├── src/
│   ├── data/               # fetching, cleaning, feature engineering
│   ├── models/             # forecasting models
│   ├── optimization/       # battery scheduling (linear programming)
│   ├── solar/              # PV generation estimation
│   ├── evaluation/         # metrics and model comparison
│   └── visualization/      # plotting utilities
├── notebooks/              # exploratory analysis and model development
├── dashboard/              # Streamlit app
├── tests/                  # pytest suite
└── reports/                # generated figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Fetch and build the dataset
python -m src.data.fetch_consumption
python -m src.data.fetch_weather
python -m src.data.build_dataset

# 2. Run the dashboard
streamlit run dashboard/app.py
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · TensorFlow · statsmodels · pvlib · PuLP · Streamlit · pytest

## License

MIT — see [LICENSE](LICENSE).
