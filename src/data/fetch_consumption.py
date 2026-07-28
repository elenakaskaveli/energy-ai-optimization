"""Download the UCI Individual Household Electric Power Consumption dataset."""

import io
import logging
import zipfile

import requests

from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_FILENAME = "household_power_consumption.txt"


def fetch_consumption(config: dict | None = None) -> None:
    config = config or load_config()
    raw_dir = resolve_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / RAW_FILENAME

    if target_path.exists():
        logger.info("Consumption data already present at %s, skipping download.", target_path)
        return

    url = config["data"]["uci_consumption_url"]
    logger.info("Downloading UCI consumption dataset from %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        member = next(name for name in zf.namelist() if name.endswith(".txt"))
        with zf.open(member) as src, open(target_path, "wb") as dst:
            dst.write(src.read())

    logger.info("Saved consumption data to %s", target_path)


if __name__ == "__main__":
    fetch_consumption()
