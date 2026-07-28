import pandas as pd
import pytest

from src.data.build_dataset import PROCESSED_FILENAME, build_dataset
from src.data.fetch_consumption import RAW_FILENAME as CONSUMPTION_FILENAME
from src.data.fetch_weather import RAW_FILENAME as WEATHER_FILENAME

CONSUMPTION_ROWS = [
    (
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;"
        "Sub_metering_1;Sub_metering_2;Sub_metering_3"
    ),
    "16/12/2006;17:00:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000",
    "16/12/2006;17:01:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000",
    "16/12/2006;18:00:00;3.520;0.500;233.290;15.800;0.000;2.000;17.000",
    "16/12/2006;18:01:00;?;0.436;233.740;23.000;0.000;1.000;17.000",
]

WEATHER_ROWS = [
    "time,temperature_2m,shortwave_radiation,wind_speed_10m,cloud_cover",
    "2006-12-16T17:00,5.0,0.0,10.0,90",
    "2006-12-16T18:00,4.5,0.0,12.0,100",
]


def _make_config(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / CONSUMPTION_FILENAME).write_text("\n".join(CONSUMPTION_ROWS))
    (raw_dir / WEATHER_FILENAME).write_text("\n".join(WEATHER_ROWS))
    return {
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "resample_freq": "1h",
        },
        "forecasting": {"target_column": "global_active_power_kw"},
    }


def test_build_dataset_merges_and_resamples(tmp_path):
    config = _make_config(tmp_path)

    merged = build_dataset(config)

    assert list(merged.index[:2]) == [
        pd.Timestamp("2006-12-16 17:00:00"),
        pd.Timestamp("2006-12-16 18:00:00"),
    ]
    assert merged.loc["2006-12-16 17:00:00", "global_active_power_kw"] == pytest.approx((4.216 + 5.360) / 2)
    assert merged["global_active_power_kw"].isna().sum() == 0
    assert "temperature_2m" in merged.columns

    output_path = tmp_path / "processed" / PROCESSED_FILENAME
    assert output_path.exists()
