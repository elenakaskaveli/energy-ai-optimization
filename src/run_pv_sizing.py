"""Recommend a rooftop PV system size that reaches a target self-sufficiency level.

Tries each candidate system size from config.yaml, estimates its generation,
and schedules the battery optimally (LP) over a sample of test-period days to
measure what fraction of demand is met without importing from the grid. This
answers the "how much PV do we actually need?" question the demand forecast
and battery optimization alone don't: those assume a fixed, arbitrary 5kW
system, while this sweeps a size range against a concrete target.
"""

import logging

import pandas as pd

from src.config import load_config, resolve_path
from src.optimization.battery import optimize_battery_lp
from src.solar.pv_estimation import estimate_pv_generation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_processed_dataset(config: dict) -> pd.DataFrame:
    path = resolve_path(config["data"]["processed_dir"]) / "household_energy_hourly.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def _self_sufficiency_pct(demand_kwh, pv_kwh, battery: dict, tariff: dict) -> float:
    """Fraction of demand covered without buying from the grid, LP-optimal battery use."""
    schedule = optimize_battery_lp(demand_kwh, pv_kwh, battery, tariff)
    total_demand = schedule["demand_kwh"].sum()
    total_grid_import = schedule["grid_import_kwh"].sum()
    return 100 * (1 - total_grid_import / total_demand)


def recommend_pv_size(raw: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    target_column = config["forecasting"]["target_column"]
    split_date = config["data"]["train_test_split_date"]
    sample_days = config["solar_pv"]["sizing_sample_days"]
    target_pct = config["solar_pv"]["target_self_sufficiency_pct"]
    battery, tariff = config["battery"], config["tariff"]

    test = raw.loc[raw.index >= split_date, [target_column]]
    test = test[test.index < test.index.max().normalize() + pd.Timedelta(days=1)]  # full days only
    full_days = [day for day, day_df in test.groupby(test.index.date) if len(day_df) == 24]
    sample_days = sorted(full_days)[:sample_days]

    results = []
    for size_kw in config["solar_pv"]["candidate_sizes_kw"]:
        pv = estimate_pv_generation(
            raw,
            latitude=config["location"]["latitude"],
            longitude=config["location"]["longitude"],
            timezone=config["location"]["timezone"],
            tilt_deg=config["solar_pv"]["tilt_deg"],
            azimuth_deg=config["solar_pv"]["azimuth_deg"],
            system_capacity_kw=size_kw,
            system_losses_pct=config["solar_pv"]["system_losses_pct"],
        )
        # Evaluate day by day (LP horizon is one day) and aggregate.
        day_scores = []
        for day in sample_days:
            day_df = raw.loc[str(day)]
            if len(day_df) != 24:
                continue
            demand = day_df[target_column].to_numpy()
            pv_kwh = pv.loc[day_df.index].to_numpy()
            day_scores.append(_self_sufficiency_pct(demand, pv_kwh, battery, tariff))
        results.append({"system_capacity_kw": size_kw, "self_sufficiency_pct": sum(day_scores) / len(day_scores)})
        logger.info("PV size %.1f kW -> %.1f%% self-sufficiency", size_kw, results[-1]["self_sufficiency_pct"])

    comparison = pd.DataFrame(results)
    meets_target = comparison[comparison["self_sufficiency_pct"] >= target_pct]
    recommended = (
        meets_target.iloc[0].to_dict()
        if not meets_target.empty
        else comparison.loc[comparison["self_sufficiency_pct"].idxmax()].to_dict()
    )
    summary = {
        "target_self_sufficiency_pct": target_pct,
        "recommended_system_capacity_kw": recommended["system_capacity_kw"],
        "recommended_self_sufficiency_pct": recommended["self_sufficiency_pct"],
        "target_met": bool(not meets_target.empty),
        "sample_days_evaluated": len(sample_days),
    }
    return summary, comparison


def main():
    config = load_config()
    raw = load_processed_dataset(config)

    summary, comparison = recommend_pv_size(raw, config)
    logger.info("PV sizing recommendation:\n%s", pd.Series(summary))

    reports_dir = resolve_path("reports")
    comparison.to_csv(reports_dir / "pv_sizing_comparison.csv", index=False)
    pd.Series(summary).to_json(reports_dir / "pv_sizing_recommendation.json", indent=2)

    return summary, comparison


if __name__ == "__main__":
    main()
