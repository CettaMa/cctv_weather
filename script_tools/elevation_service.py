"""
Terrain elevation grid service for the map elevation heatmap layer.

Source: Open-Meteo Elevation API (free, no API key), serving the Copernicus
DEM GLO-90 digital elevation model. Terrain never changes, so the sampled grid
is persisted to outputs/elevation_grid.json and reused forever; the network is
only touched when that cache is missing/stale or a refresh is requested.
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
ELEVATION_JSON_PATH = BASE_DIR / "outputs" / "elevation_grid.json"

ELEVATION_SOURCE = "Open-Meteo Elevation API (Copernicus DEM GLO-90)"

# Bandung Raya terrain window. Covers the spread of BANDUNG_WEATHER_GRID in
# open_meteo_service.py (lat -7.025..-6.815, lng 107.525..107.770) padded out to
# the basin rim: Lembang north ridge down to Soreang/Banjaran in the south.
ELEVATION_BBOX = {"north": -6.78, "south": -7.05, "west": 107.45, "east": 107.80}
ELEVATION_STEP_DEG = 0.005          # ~550 m cells -> 55 rows x 71 cols = 3905 samples
MAX_COORDS_PER_REQUEST = 100        # hard Open-Meteo limit (101 -> HTTP 400)
REQUEST_PAUSE_SECONDS = 1.0         # politeness delay between the ~40 batches (raised after HTTP 429)

ELEVATION_MEMO: Dict[str, Any] = {}


def build_grid_axes() -> Tuple[List[float], List[float]]:
    """Sample latitudes (north -> south) and longitudes (west -> east)."""
    rows = int(round((ELEVATION_BBOX["north"] - ELEVATION_BBOX["south"]) / ELEVATION_STEP_DEG)) + 1
    cols = int(round((ELEVATION_BBOX["east"] - ELEVATION_BBOX["west"]) / ELEVATION_STEP_DEG)) + 1
    lats = [round(ELEVATION_BBOX["north"] - r * ELEVATION_STEP_DEG, 5) for r in range(rows)]
    lngs = [round(ELEVATION_BBOX["west"] + c * ELEVATION_STEP_DEG, 5) for c in range(cols)]
    return lats, lngs


def _fetch_batch(lat_chunk: List[float], lng_chunk: List[float]) -> List[float]:
    url = (
        "https://api.open-meteo.com/v1/elevation?"
        f"latitude={','.join(str(v) for v in lat_chunk)}"
        f"&longitude={','.join(str(v) for v in lng_chunk)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "BandungCCTVPlatform/1.0"})
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < max_attempts:
                backoff = 5.0 * attempt
                print(f"[Elevation] HTTP 429 rate limited (attempt {attempt}/{max_attempts}); backing off {backoff:.0f}s")
                time.sleep(backoff)
                continue
            raise
    values = payload.get("elevation")
    if not isinstance(values, list) or len(values) != len(lat_chunk):
        raise ValueError(f"Open-Meteo elevation batch returned {values!r}")
    return values


def _build_payload() -> Dict[str, Any]:
    """Sample the whole grid in <=100-coordinate batches. Raises on any failure."""
    lats, lngs = build_grid_axes()
    flat_lats: List[float] = []
    flat_lngs: List[float] = []
    for lat in lats:
        for lng in lngs:
            flat_lats.append(lat)
            flat_lngs.append(lng)

    elevations: List[int] = []
    total = len(flat_lats)
    for start in range(0, total, MAX_COORDS_PER_REQUEST):
        end = min(start + MAX_COORDS_PER_REQUEST, total)
        batch = _fetch_batch(flat_lats[start:end], flat_lngs[start:end])
        elevations.extend(int(round(float(v))) for v in batch)
        print(f"[Elevation] sampled {len(elevations)}/{total} DEM points")
        if end < total:
            time.sleep(REQUEST_PAUSE_SECONDS)

    return {
        "status": "success",
        "source": ELEVATION_SOURCE,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "bbox": dict(ELEVATION_BBOX),
        "step_deg": ELEVATION_STEP_DEG,
        "rows": len(lats),
        "cols": len(lngs),
        "point_count": total,
        "min_elevation_m": min(elevations),
        "max_elevation_m": max(elevations),
        "mean_elevation_m": round(sum(elevations) / len(elevations), 1),
        "elevations": elevations,
    }


def _load_cache() -> Optional[Dict[str, Any]]:
    """Read the persisted grid, rejecting it if it no longer matches the constants."""
    if not ELEVATION_JSON_PATH.exists():
        return None
    try:
        with open(ELEVATION_JSON_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as err:
        print(f"[Elevation] cache unreadable ({err}); will refetch.")
        return None

    if data.get("status") != "success":
        return None
    if data.get("step_deg") != ELEVATION_STEP_DEG or data.get("bbox") != ELEVATION_BBOX:
        print("[Elevation] cache geometry differs from current constants; will refetch.")
        return None
    rows, cols = data.get("rows", 0), data.get("cols", 0)
    if rows * cols != len(data.get("elevations", [])) or rows == 0:
        print("[Elevation] cache is truncated; will refetch.")
        return None
    return data


def _write_cache(payload: Dict[str, Any]) -> None:
    """Atomic tmp-file + os.replace write, matching traffic_analyzer.py."""
    ELEVATION_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(ELEVATION_JSON_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        # Compact (no indent): the body is a 3905-number array.
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ELEVATION_JSON_PATH)


def fetch_elevation_grid_data(force_refresh: bool = False) -> Dict[str, Any]:
    """Return the terrain grid from memory, then disk, then the Open-Meteo API."""
    if not force_refresh:
        memo = ELEVATION_MEMO.get("grid")
        if memo:
            return memo
        cached = _load_cache()
        if cached:
            ELEVATION_MEMO["grid"] = cached
            return cached

    try:
        payload = _build_payload()
    except Exception as err:
        print(f"[Elevation] DEM fetch error: {err}. Elevation heatmap unavailable.")
        stale = ELEVATION_MEMO.get("grid") or _load_cache()
        if stale:
            return stale
        return {
            "status": "unavailable",
            "reason": str(err),
            "source": ELEVATION_SOURCE,
            "bbox": dict(ELEVATION_BBOX),
            "step_deg": ELEVATION_STEP_DEG,
            "rows": 0,
            "cols": 0,
            "point_count": 0,
            "min_elevation_m": 0,
            "max_elevation_m": 0,
            "mean_elevation_m": 0,
            "elevations": [],
        }

    _write_cache(payload)
    ELEVATION_MEMO["grid"] = payload
    return payload


if __name__ == "__main__":
    data = fetch_elevation_grid_data(force_refresh=True)
    print(
        f"[Elevation] status={data['status']} rows={data['rows']} cols={data['cols']} "
        f"points={len(data['elevations'])} min={data['min_elevation_m']}m "
        f"max={data['max_elevation_m']}m mean={data['mean_elevation_m']}m "
        f"-> {ELEVATION_JSON_PATH}"
    )
