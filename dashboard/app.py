"""Streamlit dashboard for the Smart Home Energy Management System."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.optimization.battery import heuristic_battery_schedule, no_battery_cost, optimize_battery_lp  # noqa: E402
from src.solar.pv_estimation import estimate_pv_generation  # noqa: E402

st.set_page_config(page_title="Smart Home Energy Management", layout="wide")


@st.cache_data
def load_config_cached():
    return load_config()


@st.cache_data
def load_processed_dataset(_config: dict) -> pd.DataFrame:
    path = resolve_path(_config["data"]["processed_dir"]) / "household_energy_hourly.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


@st.cache_data
def load_pv_series(_config: dict, _raw: pd.DataFrame) -> pd.Series:
    return estimate_pv_generation(
        _raw,
        latitude=_config["location"]["latitude"],
        longitude=_config["location"]["longitude"],
        timezone=_config["location"]["timezone"],
        tilt_deg=_config["solar_pv"]["tilt_deg"],
        azimuth_deg=_config["solar_pv"]["azimuth_deg"],
        system_capacity_kw=_config["solar_pv"]["system_capacity_kw"],
        system_losses_pct=_config["solar_pv"]["system_losses_pct"],
    )


@st.cache_data
def load_forecast_predictions() -> pd.DataFrame | None:
    path = resolve_path("reports") / "forecasting_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    path = resolve_path("reports") / "forecasting_model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


config = load_config_cached()
raw = load_processed_dataset(config)
pv_series = load_pv_series(config, raw)
target_column = config["forecasting"]["target_column"]
split_date = config["data"]["train_test_split_date"]

st.title("🏠 Smart Home Energy Management System")
st.caption(
    "AI-driven demand forecasting, battery optimization, and solar PV integration "
    "for a single household — built on real consumption and weather data."
)

tab_forecast, tab_battery, tab_models = st.tabs(
    ["📈 Demand Forecast", "🔋 Battery & Solar", "📊 Model Comparison"]
)

# ---------------------------------------------------------------------------
# Tab 1: Demand forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    predictions = load_forecast_predictions()
    if predictions is None:
        st.warning("Run `python -m src.run_forecasting` first to generate forecast results.")
    else:
        min_date, max_date = predictions.index.min().date(), predictions.index.max().date()
        selected_date = st.date_input(
            "Select a day to inspect", value=min_date, min_value=min_date, max_value=max_date, key="forecast_date"
        )
        day_predictions = predictions.loc[str(selected_date)]

        available_models = [c for c in predictions.columns if c != "actual"]
        chosen_models = st.multiselect(
            "Models to display", available_models, default=["XGBoost (day-ahead)", "LSTM"]
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=day_predictions.index, y=day_predictions["actual"], name="Actual", line=dict(color="black", width=3)))
        for model in chosen_models:
            fig.add_trace(go.Scatter(x=day_predictions.index, y=day_predictions[model], name=model))
        fig.update_layout(
            title=f"Household demand — {selected_date}",
            yaxis_title="Global active power (kW)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Battery & solar
# ---------------------------------------------------------------------------
with tab_battery:
    test_dates = raw.loc[raw.index >= split_date].index
    min_date, max_date = test_dates.min().date(), test_dates.max().date()
    selected_date = st.date_input(
        "Select a day to schedule", value=pd.Timestamp("2010-07-15").date(), min_value=min_date, max_value=max_date, key="battery_date"
    )

    col1, col2, col3 = st.columns(3)
    battery_capacity = col1.slider("Battery capacity (kWh)", 2.0, 20.0, float(config["battery"]["capacity_kwh"]), step=1.0)
    pv_capacity = col2.slider("PV system size (kWp)", 1.0, 10.0, float(config["solar_pv"]["system_capacity_kw"]), step=0.5)
    peak_price = col3.slider("Peak price (EUR/kWh)", 0.10, 0.60, float(config["tariff"]["peak_price_per_kwh"]), step=0.01)

    battery_cfg = dict(config["battery"])
    battery_cfg["capacity_kwh"] = battery_capacity
    tariff_cfg = dict(config["tariff"])
    tariff_cfg["peak_price_per_kwh"] = peak_price

    day_df = raw.loc[str(selected_date), [target_column]].copy()
    pv_scale = pv_capacity / config["solar_pv"]["system_capacity_kw"]
    day_pv = (pv_series.loc[str(selected_date)] * pv_scale).to_numpy()
    day_demand = day_df[target_column].to_numpy()

    if len(day_demand) == 24:
        no_batt = no_battery_cost(day_demand, day_pv, tariff_cfg)
        heuristic = heuristic_battery_schedule(day_demand, day_pv, battery_cfg, tariff_cfg)
        lp = optimize_battery_lp(day_demand, day_pv, battery_cfg, tariff_cfg)

        m1, m2, m3 = st.columns(3)
        m1.metric("No battery cost", f"€{no_batt:.2f}")
        m2.metric("Heuristic cost", f"€{heuristic['cost'].sum():.2f}", f"-{100 * (1 - heuristic['cost'].sum() / no_batt):.0f}%")
        m3.metric("LP-optimal cost", f"€{lp['cost'].sum():.2f}", f"-{100 * (1 - lp['cost'].sum() / no_batt):.0f}%")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=day_df.index, y=day_demand, name="Demand", line=dict(color="black")))
        fig1.add_trace(go.Scatter(x=day_df.index, y=day_pv, name="PV generation", line=dict(color="orange")))
        fig1.update_layout(title="Demand vs. solar PV generation", yaxis_title="kW", height=350)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=day_df.index, y=lp["soc_kwh"], name="SOC (LP-optimal)", line=dict(color="green")))
        fig2.add_trace(go.Scatter(x=day_df.index, y=heuristic["soc_kwh"], name="SOC (heuristic)", line=dict(color="green", dash="dash")))
        fig2.update_layout(title="Battery state of charge", yaxis_title="kWh", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Selected day doesn't have a full 24 hours of data (likely a DST transition).")

# ---------------------------------------------------------------------------
# Tab 3: Model comparison
# ---------------------------------------------------------------------------
with tab_models:
    comparison = load_model_comparison()
    if comparison is None:
        st.warning("Run `python -m src.run_forecasting` first to generate model comparison results.")
    else:
        st.subheader("Demand forecasting: model comparison")
        st.dataframe(comparison.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="#c6f6c6"))

        shap_path = resolve_path("reports/figures") / "shap_summary.png"
        if shap_path.exists():
            st.subheader("Feature importance (SHAP) — XGBoost day-ahead model")
            st.image(str(shap_path))
