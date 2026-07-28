import numpy as np
import pytest

from src.optimization.battery import heuristic_battery_schedule, no_battery_cost, optimize_battery_lp

BATTERY = {
    "capacity_kwh": 10.0,
    "max_charge_rate_kw": 5.0,
    "max_discharge_rate_kw": 5.0,
    "round_trip_efficiency": 0.9,
    "min_soc_pct": 10,
    "max_soc_pct": 100,
    "initial_soc_pct": 50,
}
TARIFF = {
    "peak_price_per_kwh": 0.30,
    "off_peak_price_per_kwh": 0.15,
    "peak_hours": [8, 9, 10, 11, 18, 19, 20, 21],
}


def _sample_profiles(n=24, seed=0):
    rng = np.random.default_rng(seed)
    demand = rng.uniform(0.3, 2.5, n)
    pv = np.clip(np.sin(np.linspace(0, np.pi, n)) * 3, 0, None)
    return demand, pv


def test_lp_schedule_respects_soc_bounds():
    demand, pv = _sample_profiles()
    schedule = optimize_battery_lp(demand, pv, BATTERY, TARIFF)

    min_soc = BATTERY["min_soc_pct"] / 100 * BATTERY["capacity_kwh"]
    max_soc = BATTERY["max_soc_pct"] / 100 * BATTERY["capacity_kwh"]
    assert (schedule["soc_kwh"] >= min_soc - 1e-6).all()
    assert (schedule["soc_kwh"] <= max_soc + 1e-6).all()


def test_lp_schedule_respects_rate_limits():
    demand, pv = _sample_profiles()
    schedule = optimize_battery_lp(demand, pv, BATTERY, TARIFF)
    assert (schedule["charge_kwh"] <= BATTERY["max_charge_rate_kw"] + 1e-6).all()
    assert (schedule["discharge_kwh"] <= BATTERY["max_discharge_rate_kw"] + 1e-6).all()


def test_lp_schedule_satisfies_energy_balance():
    demand, pv = _sample_profiles()
    schedule = optimize_battery_lp(demand, pv, BATTERY, TARIFF)
    supply = schedule["pv_kwh"] + schedule["discharge_kwh"] + schedule["grid_import_kwh"]
    use = schedule["demand_kwh"] + schedule["charge_kwh"] + schedule["grid_export_kwh"]
    np.testing.assert_allclose(supply, use, atol=1e-6)


def test_lp_optimizer_is_at_least_as_good_as_heuristic():
    demand, pv = _sample_profiles(n=48, seed=1)
    lp_schedule = optimize_battery_lp(demand, pv, BATTERY, TARIFF)
    heuristic_schedule = heuristic_battery_schedule(demand, pv, BATTERY, TARIFF)

    lp_cost = lp_schedule["cost"].sum()
    heuristic_cost = heuristic_schedule["cost"].sum()
    # LP finds the true optimum, so it can never do worse than the greedy heuristic
    assert lp_cost <= heuristic_cost + 1e-6


def test_no_battery_cost_matches_manual_calculation():
    demand = np.array([1.0, 2.0])
    pv = np.array([0.5, 3.0])
    tariff = {"peak_price_per_kwh": 0.30, "off_peak_price_per_kwh": 0.15, "peak_hours": []}
    # hour 0: residual = 0.5 import @ 0.15; hour 1: residual = -1.0 export @ 0.06 credit
    expected = 0.5 * 0.15 - 1.0 * 0.06
    assert no_battery_cost(demand, pv, tariff, feed_in_price_per_kwh=0.06) == pytest.approx(expected)
