"""Temperature grid service (placeholder, forecast-ready contract).

Provides a 24-hour hourly temperature grid over Greater Bandung derived from
the Copernicus DEM elevation grid via lapse rate, a diurnal temperature cycle,
and an urban heat island (UHI) centered on Alun-Alun Bandung.

Designed to be swapped cleanly with numerical weather prediction / forecasting
output once available without changing the API schema.
"""

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional

from script_tools.elevation_service import (
    build_grid_axes,
    fetch_elevation_grid_data,
)

TEMPERATURE_SOURCE = (
    "PLACEHOLDER — lapse rate 6,5 °C/km dari DEM Copernicus GLO-90 + "
    "siklus harian; akan diganti hasil forecasting"
)

# Physical lapse-rate and climate baseline for Bandung basin
SEA_LEVEL_MEAN_C = 31.0
LAPSE_RATE_C_PER_M = 0.0065
DIURNAL_AMPLITUDE_C = 4.0
DIURNAL_PEAK_HOUR = 14

# Urban heat island (Alun-Alun Bandung)
UHI_CENTER = (-6.9210, 107.6070)
UHI_PEAK_C = 1.5
UHI_SIGMA_DEG = 0.04

FORECAST_HOURS = 24
WIB = timezone(timedelta(hours=7))

# In-memory memo keyed by hour start ISO string
TEMPERATURE_MEMO: Dict[str, Any] = {}


def _build_placeholder_payload(
    elev: Dict[str, Any], start: datetime
) -> Dict[str, Any]:
    lats, lngs = build_grid_axes()
    rows = elev["rows"]
    cols = elev["cols"]
    elevations = elev["elevations"]

    uhi_lat, uhi_lng = UHI_CENTER
    two_sigma_sq = 2.0 * (UHI_SIGMA_DEG ** 2)

    # Precompute per-cell baseline (elevation + UHI)
    cell_baselines: List[Optional[float]] = []
    for r in range(rows):
        lat = lats[r]
        d_lat_sq = (lat - uhi_lat) ** 2
        for c in range(cols):
            lng = lngs[c]
            idx = r * cols + c
            elev_m = elevations[idx]
            if elev_m is None:
                cell_baselines.append(None)
                continue
            d_sq = d_lat_sq + (lng - uhi_lng) ** 2
            uhi_effect = UHI_PEAK_C * math.exp(-d_sq / two_sigma_sq)
            base_temp = SEA_LEVEL_MEAN_C - (LAPSE_RATE_C_PER_M * elev_m) + uhi_effect
            cell_baselines.append(base_temp)

    times: List[str] = []
    frames: List[List[Optional[float]]] = []
    min_temp = float("inf")
    max_temp = float("-inf")

    total_cells = rows * cols
    for k in range(FORECAST_HOURS):
        frame_time = start + timedelta(hours=k)
        times.append(frame_time.isoformat(timespec="minutes"))
        hour = frame_time.hour
        diurnal = DIURNAL_AMPLITUDE_C * math.cos(
            2.0 * math.pi * (hour - DIURNAL_PEAK_HOUR) / 24.0
        )

        frame_values: List[Optional[float]] = [None] * total_cells
        for i, base in enumerate(cell_baselines):
            if base is None:
                continue
            val = round(base + diurnal, 1)
            frame_values[i] = val
            if val < min_temp:
                min_temp = val
            if val > max_temp:
                max_temp = val
        frames.append(frame_values)

    return {
        "status": "success",
        "is_placeholder": True,
        "source": TEMPERATURE_SOURCE,
        "generated_at": datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S"),
        "bbox": elev["bbox"],
        "step_deg": elev["step_deg"],
        "rows": rows,
        "cols": cols,
        "unit": "°C",
        "times": times,
        "min_temperature_c": None if min_temp == float("inf") else min_temp,
        "max_temperature_c": None if max_temp == float("-inf") else max_temp,
        "frames": frames,
    }


def fetch_temperature_grid_data() -> Dict[str, Any]:
    """Return 24-hour hourly temperature grid over Bandung.

    Memoizes result keyed by the current WIB hour start so subsequent calls
    in the same hour are instant.
    """
    start = datetime.now(WIB).replace(minute=0, second=0, microsecond=0)
    cache_key = start.isoformat()
    if cache_key in TEMPERATURE_MEMO:
        return TEMPERATURE_MEMO[cache_key]

    elev = fetch_elevation_grid_data()
    if elev.get("status") != "success" or not elev.get("elevations"):
        return {
            "status": "unavailable",
            "reason": elev.get("reason", "Grid DEM tidak tersedia"),
            "is_placeholder": True,
            "source": TEMPERATURE_SOURCE,
            "times": [],
            "frames": [],
        }

    payload = _build_placeholder_payload(elev, start)
    TEMPERATURE_MEMO.clear()
    TEMPERATURE_MEMO[cache_key] = payload
    return payload


if __name__ == "__main__":
    data = fetch_temperature_grid_data()
    print(f"Status: {data.get('status')}")
    print(f"Grid: {data.get('rows')} rows x {data.get('cols')} cols")
    print(f"Times: {len(data.get('times', []))} frames, first={data.get('times', [None])[0]}")
    print(f"Min temp: {data.get('min_temperature_c')} °C, Max temp: {data.get('max_temperature_c')} °C")
