"""Estimate rooftop solar PV generation from historical weather using pvlib (PVWatts model)."""

import pandas as pd
import pvlib


def estimate_pv_generation(
    weather: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone: str,
    tilt_deg: float,
    azimuth_deg: float,
    system_capacity_kw: float,
    system_losses_pct: float,
) -> pd.Series:
    """Estimate AC power output (kW) of a rooftop PV system from GHI + weather.

    `weather` must have a DatetimeIndex and columns `shortwave_radiation` (GHI,
    W/m^2), `temperature_2m` (°C), and `wind_speed_10m` (m/s).
    """
    local_index = weather.index.tz_localize(timezone, ambiguous=True, nonexistent="shift_forward")
    ghi = weather["shortwave_radiation"].to_numpy()  # global horizontal irradiance, W/m^2

    # Where the sun is in the sky at each hour (pure geometry from location + time).
    solar_position = pvlib.solarposition.get_solarposition(local_index, latitude, longitude)

    # We only measured total irradiance on a horizontal surface (GHI); split it into
    # its direct-beam (DNI) and diffuse-sky (DHI) components so it can be projected
    # onto the panel's actual tilt in the next step.
    erbs_out = pvlib.irradiance.erbs(ghi, solar_position["zenith"].to_numpy(), local_index)
    dni = erbs_out["dni"]
    dhi = erbs_out["dhi"]

    # Irradiance actually falling on the tilted, oriented panel surface (plane-of-array).
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
    )

    # Panels lose efficiency as they heat up, so estimate cell temperature from
    # ambient air temperature, wind (cooling), and the irradiance hitting them.
    cell_temperature = pvlib.temperature.faiman(
        poa["poa_global"], weather["temperature_2m"].to_numpy(), weather["wind_speed_10m"].to_numpy()
    )

    # PVWatts (NREL) DC output model: roughly proportional to irradiance, with a
    # small downward correction (gamma_pdc) for cell temperature above 25°C.
    pdc0_w = system_capacity_kw * 1000
    dc_power_w = pvlib.pvsystem.pvwatts_dc(poa["poa_global"], cell_temperature, pdc0=pdc0_w, gamma_pdc=-0.004)
    ac_power_w = pvlib.inverter.pvwatts(dc_power_w, pdc0=pdc0_w)  # DC -> AC via inverter efficiency curve

    # Fixed real-world losses (wiring, soiling, mismatch) not captured by the physical
    # model above, plus clean-up: no negative output, and no gaps (e.g. at night).
    losses_factor = 1 - system_losses_pct / 100
    ac_power_kw = (ac_power_w / 1000) * losses_factor
    ac_power_kw = ac_power_kw.clip(lower=0).fillna(0)
    ac_power_kw.index = weather.index
    return ac_power_kw.rename("pv_generation_kw")
