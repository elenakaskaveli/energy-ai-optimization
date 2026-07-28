"""Estimate rooftop solar PV generation and evaluate battery scheduling savings.

Compares three scenarios over every day of the test period: no battery
(PV self-consumption only), a greedy rule-based battery heuristic, and the
LP-optimal battery schedule.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, resolve_path
from src.optimization.battery import heuristic_battery_schedule, no_battery_cost, optimize_battery_lp
from src.solar.pv_estimation import estimate_pv_generation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_processed_dataset(config: dict) -> pd.DataFrame:
    path = resolve_path(config["data"]["processed_dir"]) / "household_energy_hourly.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def main():
    config = load_config()
    target_column = config["forecasting"]["target_column"]
    split_date = config["data"]["train_test_split_date"]

    raw = load_processed_dataset(config)
    pv = estimate_pv_generation(
        raw,
        latitude=config["location"]["latitude"],
        longitude=config["location"]["longitude"],
        timezone=config["location"]["timezone"],
        tilt_deg=config["solar_pv"]["tilt_deg"],
        azimuth_deg=config["solar_pv"]["azimuth_deg"],
        system_capacity_kw=config["solar_pv"]["system_capacity_kw"],
        system_losses_pct=config["solar_pv"]["system_losses_pct"],
    )

    test = raw.loc[raw.index >= split_date, [target_column]].join(pv)
    test = test[test.index < test.index.max().normalize() + pd.Timedelta(days=1)]  # keep full days only

    battery, tariff = config["battery"], config["tariff"]

    daily_totals = []
    example_day_schedules = None
    for day, day_df in test.groupby(test.index.date):
        if len(day_df) != 24:
            continue  # skip partial days (e.g. DST transitions)

        demand = day_df[target_column].to_numpy()
        pv_kwh = day_df["pv_generation_kw"].to_numpy()

        no_batt_cost = no_battery_cost(demand, pv_kwh, tariff)
        heuristic = heuristic_battery_schedule(demand, pv_kwh, battery, tariff)
        lp = optimize_battery_lp(demand, pv_kwh, battery, tariff)

        daily_totals.append(
            {
                "date": day,
                "no_battery_cost": no_batt_cost,
                "heuristic_cost": heuristic["cost"].sum(),
                "lp_cost": lp["cost"].sum(),
            }
        )

        if str(day) == "2010-07-15":
            example_day_schedules = {"heuristic": heuristic, "lp": lp, "index": day_df.index}

    totals = pd.DataFrame(daily_totals)
    summary = {
        "days_evaluated": len(totals),
        "no_battery_total_eur": totals["no_battery_cost"].sum(),
        "heuristic_total_eur": totals["heuristic_cost"].sum(),
        "lp_total_eur": totals["lp_cost"].sum(),
        "heuristic_savings_pct": 100 * (1 - totals["heuristic_cost"].sum() / totals["no_battery_cost"].sum()),
        "lp_savings_pct": 100 * (1 - totals["lp_cost"].sum() / totals["no_battery_cost"].sum()),
        "lp_vs_heuristic_improvement_pct": 100
        * (1 - totals["lp_cost"].sum() / totals["heuristic_cost"].sum()),
    }
    logger.info("Battery optimization summary over %d days:\n%s", len(totals), pd.Series(summary))

    reports_dir = resolve_path("reports")
    totals.to_csv(reports_dir / "battery_daily_costs.csv", index=False)
    pd.Series(summary).to_json(reports_dir / "battery_optimization_summary.json", indent=2)

    if example_day_schedules is not None:
        _plot_example_day(
            example_day_schedules, resolve_path("reports/figures") / "battery_schedule_example.png"
        )

    return summary, totals


def _plot_example_day(schedules: dict, output_path):
    idx = schedules["index"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(idx, schedules["lp"]["demand_kwh"], label="Demand", color="black")
    axes[0].plot(idx, schedules["lp"]["pv_kwh"], label="PV generation", color="orange")
    axes[0].set_ylabel("kW")
    axes[0].set_title("Household demand vs. solar PV generation — 2010-07-15")
    axes[0].legend()

    axes[1].plot(idx, schedules["lp"]["soc_kwh"], label="Battery SOC (LP-optimal)", color="green")
    axes[1].plot(
        idx, schedules["heuristic"]["soc_kwh"], label="Battery SOC (heuristic)", color="green", linestyle="--"
    )
    axes[1].set_ylabel("State of charge (kWh)")
    axes[1].set_title("Battery state of charge: LP-optimal vs. rule-based heuristic")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    logger.info("Saved battery schedule example plot to %s", output_path)


if __name__ == "__main__":
    main()
