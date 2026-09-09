import numpy as np

from src.run_pv_sizing import _self_sufficiency_pct

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


def test_zero_pv_gives_low_self_sufficiency():
    # Only the battery's pre-existing charge (initial_soc_pct) can help here,
    # so self-sufficiency should be small but not exactly zero.
    demand = np.full(24, 1.0)
    pv = np.zeros(24)
    result = _self_sufficiency_pct(demand, pv, BATTERY, TARIFF)
    assert 0.0 < result < 25.0


def test_ample_pv_and_battery_gives_high_self_sufficiency():
    demand = np.full(24, 1.0)
    pv = np.full(24, 3.0)  # far more than demand every hour
    assert _self_sufficiency_pct(demand, pv, BATTERY, TARIFF) > 95.0
