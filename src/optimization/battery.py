"""Battery charge/discharge scheduling: LP optimizer and a rule-based heuristic baseline."""

import math

import numpy as np
import pandas as pd
import pulp


def _hourly_prices(n_hours: int, tariff: dict) -> np.ndarray:
    peak_hours = set(tariff["peak_hours"])
    return np.array(
        [
            tariff["peak_price_per_kwh"] if (h % 24) in peak_hours else tariff["off_peak_price_per_kwh"]
            for h in range(n_hours)
        ]
    )


def optimize_battery_lp(
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    battery: dict,
    tariff: dict,
    feed_in_price_per_kwh: float = 0.06,
    cycling_cost_per_kwh: float = 0.002,
) -> pd.DataFrame:
    """Minimize grid electricity cost over the horizon by scheduling battery charge/discharge.

    `demand_kwh` and `pv_kwh` are hourly arrays (1 value = 1 kWh at hourly resolution).
    `cycling_cost_per_kwh` is a small notional cost per kWh charged/discharged
    (representing battery wear); it is far smaller than any electricity price
    and mainly serves as a tie-breaker so the optimizer doesn't cycle the
    battery when doing so has no effect on total grid cost.
    Returns a DataFrame with the optimal schedule and per-hour cost.
    """
    n = len(demand_kwh)
    prices = _hourly_prices(n, tariff)

    capacity = battery["capacity_kwh"]
    max_charge = battery["max_charge_rate_kw"]
    max_discharge = battery["max_discharge_rate_kw"]
    eta = math.sqrt(battery["round_trip_efficiency"])  # split efficiency between charge and discharge
    min_soc = battery["min_soc_pct"] / 100 * capacity
    max_soc = battery["max_soc_pct"] / 100 * capacity
    initial_soc = battery["initial_soc_pct"] / 100 * capacity

    prob = pulp.LpProblem("battery_schedule", pulp.LpMinimize)

    charge = [pulp.LpVariable(f"charge_{t}", 0, max_charge) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", 0, max_discharge) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", min_soc, max_soc) for t in range(n)]
    grid_import = [pulp.LpVariable(f"grid_import_{t}", 0) for t in range(n)]
    grid_export = [pulp.LpVariable(f"grid_export_{t}", 0) for t in range(n)]

    for t in range(n):
        prev_soc = initial_soc if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + charge[t] * eta - discharge[t] / eta
        # energy balance: supply == demand
        prob += pv_kwh[t] + discharge[t] + grid_import[t] == demand_kwh[t] + charge[t] + grid_export[t]

    prob += pulp.lpSum(
        grid_import[t] * prices[t]
        - grid_export[t] * feed_in_price_per_kwh
        + (charge[t] + discharge[t]) * cycling_cost_per_kwh
        for t in range(n)
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return pd.DataFrame(
        {
            "demand_kwh": demand_kwh,
            "pv_kwh": pv_kwh,
            "charge_kwh": [v.value() for v in charge],
            "discharge_kwh": [v.value() for v in discharge],
            "soc_kwh": [v.value() for v in soc],
            "grid_import_kwh": [v.value() for v in grid_import],
            "grid_export_kwh": [v.value() for v in grid_export],
            "price_per_kwh": prices,
            "cost": [
                grid_import[t].value() * prices[t] - grid_export[t].value() * feed_in_price_per_kwh
                for t in range(n)
            ],
        }
    )


def heuristic_battery_schedule(
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    battery: dict,
    tariff: dict,
    feed_in_price_per_kwh: float = 0.06,
) -> pd.DataFrame:
    """Greedy rule: charge from any PV surplus, discharge to cover any PV deficit."""
    n = len(demand_kwh)
    prices = _hourly_prices(n, tariff)

    capacity = battery["capacity_kwh"]
    max_charge = battery["max_charge_rate_kw"]
    max_discharge = battery["max_discharge_rate_kw"]
    eta = math.sqrt(battery["round_trip_efficiency"])
    min_soc = battery["min_soc_pct"] / 100 * capacity
    max_soc = battery["max_soc_pct"] / 100 * capacity
    soc = battery["initial_soc_pct"] / 100 * capacity

    rows = []
    for t in range(n):
        net = pv_kwh[t] - demand_kwh[t]
        charge = discharge = 0.0

        if net > 0:
            room = (max_soc - soc) / eta
            charge = min(net, max_charge, room)
        elif net < 0:
            available = (soc - min_soc) * eta
            discharge = min(-net, max_discharge, available)

        soc = soc + charge * eta - discharge / eta

        residual = demand_kwh[t] + charge - pv_kwh[t] - discharge
        grid_import = max(residual, 0.0)
        grid_export = max(-residual, 0.0)

        rows.append(
            {
                "demand_kwh": demand_kwh[t],
                "pv_kwh": pv_kwh[t],
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "soc_kwh": soc,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "price_per_kwh": prices[t],
                "cost": grid_import * prices[t] - grid_export * feed_in_price_per_kwh,
            }
        )

    return pd.DataFrame(rows)


def no_battery_cost(
    demand_kwh: np.ndarray, pv_kwh: np.ndarray, tariff: dict, feed_in_price_per_kwh: float = 0.06
) -> float:
    """Reference cost if there were no battery at all (PV self-consumption only)."""
    n = len(demand_kwh)
    prices = _hourly_prices(n, tariff)
    residual = demand_kwh - pv_kwh
    grid_import = np.clip(residual, 0, None)
    grid_export = np.clip(-residual, 0, None)
    return float(np.sum(grid_import * prices - grid_export * feed_in_price_per_kwh))
