#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Meteo Integration Service for Bandung CCTV & Weather Platform.
Integrates:
1. 🌧️ Real-Time Precipitation & Weather Grid (Precipitation mm/h, Temp, Humidity, WMO codes).
2. 💨 Wind Field & Vector Streamlines (Wind speed km/h, Wind direction degrees, Gusts).
3. 🌊 Hydrology & River Discharge Forecast (GloFAS Flood API m³/s for major Bandung rivers).
"""

import math
import time
import json
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

# In-memory cache for Open-Meteo responses with TTL
WEATHER_CACHE: Dict[str, Any] = {}
HYDRO_CACHE: Dict[str, Any] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes cache

# 12 Strategic Weather Grid Monitoring Stations across Greater Bandung
BANDUNG_WEATHER_GRID = [
    {"id": "grid_alun_alun", "name": "Bandung Pusat (Alun-Alun / Asia Afrika)", "lat": -6.9210, "lng": 107.6070, "zone": "Pusat"},
    {"id": "grid_dago", "name": "Bandung Utara (Dago / Dipatiukur)", "lat": -6.8850, "lng": 107.6150, "zone": "Utara"},
    {"id": "grid_lembang", "name": "Lembang Highland (Bandung Utara Atas)", "lat": -6.8150, "lng": 107.6200, "zone": "Utara"},
    {"id": "grid_pasteur", "name": "Bandung Barat (Pasteur / Sukajadi)", "lat": -6.8950, "lng": 107.5750, "zone": "Barat"},
    {"id": "grid_cimahi", "name": "Cimahi Kota (Barat)", "lat": -6.8850, "lng": 107.5400, "zone": "Barat"},
    {"id": "grid_kopo", "name": "Bandung Selatan (Kopo / Pasirkoja)", "lat": -6.9500, "lng": 107.5850, "zone": "Selatan"},
    {"id": "grid_dayeuhkolot", "name": "Dayeuhkolot / Baleendah (Citarum)", "lat": -6.9850, "lng": 107.6250, "zone": "Selatan"},
    {"id": "grid_gedebage", "name": "Bandung Timur (Gedebage / Derwati)", "lat": -6.9550, "lng": 107.6850, "zone": "Timur"},
    {"id": "grid_arcamanik", "name": "Bandung Timur (Arcamanik / Cingised)", "lat": -6.9150, "lng": 107.6750, "zone": "Timur"},
    {"id": "grid_cibiru", "name": "Ujungberung / Cibiru (Timur)", "lat": -6.9100, "lng": 107.7100, "zone": "Timur"},
    {"id": "grid_soreang", "name": "Soreang / Kutawaringin", "lat": -7.0250, "lng": 107.5250, "zone": "Selatan"},
    {"id": "grid_jatinangor", "name": "Jatinangor (Timur)", "lat": -6.9300, "lng": 107.7700, "zone": "Timur"}
]

# 5 Major River Hydrology Stations (GloFAS Flood API)
BANDUNG_RIVER_STATIONS = [
    {
        "id": "river_citarum_dayeuhkolot",
        "name": "Sungai Citarum (Dayeuhkolot / Baleendah)",
        "river": "Citarum",
        "lat": -6.9850,
        "lng": 107.6250,
        "normal_discharge": 12.0,
        "warning_threshold": 35.0,
        "danger_threshold": 60.0,
        "drainage_area": "Cekungan Bandung Utama",
        "description": "Titik temu saluran drainase se-Bandung Raya sebelum dialirkan ke Waduk Saguling."
    },
    {
        "id": "river_citepus_pasteur",
        "name": "Sungai Citepus (Pasteur / Pagarsih)",
        "river": "Citepus",
        "lat": -6.8950,
        "lng": 107.5750,
        "normal_discharge": 4.5,
        "warning_threshold": 12.0,
        "danger_threshold": 22.0,
        "drainage_area": "Pasteur, Sukajadi, Pagarsih",
        "description": "Saluran kritis yang melintasi Underpass Pasteur, Pagarsih, dan pusat perbelanjaan BTC."
    },
    {
        "id": "river_cikapundung_dago",
        "name": "Sungai Cikapundung (Dago / Alun-Alun)",
        "river": "Cikapundung",
        "lat": -6.8850,
        "lng": 107.6150,
        "normal_discharge": 6.0,
        "warning_threshold": 18.0,
        "danger_threshold": 32.0,
        "drainage_area": "Dago, Asia Afrika, Alun-Alun",
        "description": "Sungai utama yang membelah Kota Bandung dari Curug Dago hingga muara Citarum."
    },
    {
        "id": "river_cinambo_gedebage",
        "name": "Sungai Cinambo (Gedebage / Derwati)",
        "river": "Cinambo",
        "lat": -6.9550,
        "lng": 107.6850,
        "normal_discharge": 3.5,
        "warning_threshold": 10.0,
        "danger_threshold": 18.0,
        "drainage_area": "Gedebage, Derwati, GBLA",
        "description": "Saluran drainase dataran rendah Bandung Timur yang rentan genangan saat Citarum pasang."
    },
    {
        "id": "river_cipamokolan_cingised",
        "name": "Sungai Cipamokolan (Arcamanik / Cingised)",
        "river": "Cipamokolan",
        "lat": -6.9150,
        "lng": 107.6750,
        "normal_discharge": 3.0,
        "warning_threshold": 9.0,
        "danger_threshold": 16.0,
        "drainage_area": "Cingised, Cisaranten, Arcamanik",
        "description": "Saluran aliran pembuangan Arcamanik yang sering meluap ke badan jalan Cingised."
    }
]

# WMO Weather Code Translations & Icons
WMO_WEATHER_MAP = {
    0: ("Cerah", "☀️", "#F59E0B"),
    1: ("Sebagian Cerah", "🌤️", "#FBBF24"),
    2: ("Cerah Berawan", "⛅", "#93C5FD"),
    3: ("Berawan Tebal", "☁️", "#94A3B8"),
    45: ("Berkabut", "🌫️", "#64748B"),
    48: ("Kabut Tebal Berembun", "🌫️", "#64748B"),
    51: ("Gerimis Sangat Ringan", "🌦️", "#38BDF8"),
    53: ("Gerimis Ringan", "🌦️", "#0EA5E9"),
    55: ("Gerimis Lebat", "🌧️", "#0284C7"),
    61: ("Hujan Ringan", "🌧️", "#10B981"),
    63: ("Hujan Sedang", "🌧️", "#3B82F6"),
    65: ("Hujan Lebat", "🌧️", "#1D4ED8"),
    80: ("Hujan Lokal Ringan", "🌦️", "#06B6D4"),
    81: ("Hujan Lokal Sedang", "🌧️", "#0284C7"),
    82: ("Hujan Lokal Sangat Deras", "⛈️", "#7C3AED"),
    95: ("Badai Petir Ringan/Sedang", "⛈️", "#EF4444"),
    96: ("Badai Petir Disertai Butiran Es", "⛈️", "#DC2626"),
    99: ("Badai Petir Ekstrem", "⛈️", "#991B1B")
}


def decode_wmo_weather(code: int):
    """Translate WMO weather code to Indonesian text, emoji, and color."""
    return WMO_WEATHER_MAP.get(code, ("Berawan", "⛅", "#3B82F6"))


def get_precipitation_category(precip_mm: float):
    """Categorize precipitation intensity in mm/hour."""
    if precip_mm < 0.1:
        return {"category": "Cerah / Kering", "level": "NONE", "color": "transparent", "intensity": 0.0}
    elif precip_mm < 2.5:
        return {"category": "Hujan Ringan / Gerimis", "level": "LIGHT", "color": "#10B981", "intensity": 0.3}
    elif precip_mm < 10.0:
        return {"category": "Hujan Sedang", "level": "MODERATE", "color": "#F59E0B", "intensity": 0.6}
    elif precip_mm < 20.0:
        return {"category": "Hujan Lebat", "level": "HEAVY", "color": "#EF4444", "intensity": 0.85}
    else:
        return {"category": "Hujan Sangat Lebat / Ekstrem", "level": "EXTREME", "color": "#7C3AED", "intensity": 1.0}


def fetch_weather_grid_data() -> Dict[str, Any]:
    """
    Fetch multi-point real-time weather & precipitation data across Bandung Raya from Open-Meteo.
    Uses in-memory cache to respect rate limits.
    """
    now = time.time()
    cached = WEATHER_CACHE.get("grid")
    if cached and (now - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]

    lats = [str(pt["lat"]) for pt in BANDUNG_WEATHER_GRID]
    lngs = [str(pt["lng"]) for pt in BANDUNG_WEATHER_GRID]

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={','.join(lats)}&longitude={','.join(lngs)}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        f"&timezone=Asia%2FJakarta"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BandungCCTVPlatform/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_data = json.loads(resp.read().decode("utf-8"))

        # If multiple points, Open-Meteo returns a list of responses
        results_list = raw_data if isinstance(raw_data, list) else [raw_data]

        grid_output = []
        wind_vectors = []
        max_precip = 0.0

        for idx, grid_meta in enumerate(BANDUNG_WEATHER_GRID):
            cur = {}
            if idx < len(results_list):
                cur = results_list[idx].get("current", {})

            temp = cur.get("temperature_2m", 27.0)
            humidity = cur.get("relative_humidity_2m", 70)
            precip = cur.get("precipitation", 0.0)
            wcode = cur.get("weather_code", 2)
            wspeed = cur.get("wind_speed_10m", 8.0)
            wdir = cur.get("wind_direction_10m", 220)
            wgusts = cur.get("wind_gusts_10m", 15.0)

            if precip > max_precip:
                max_precip = precip

            w_text, w_icon, w_color = decode_wmo_weather(wcode)
            p_info = get_precipitation_category(precip)

            # Wind vector components (u: west-east, v: south-north in m/s)
            rad = math.radians(wdir)
            speed_ms = wspeed / 3.6
            u_comp = -speed_ms * math.sin(rad)
            v_comp = -speed_ms * math.cos(rad)

            point_data = {
                "id": grid_meta["id"],
                "name": grid_meta["name"],
                "zone": grid_meta["zone"],
                "lat": grid_meta["lat"],
                "lng": grid_meta["lng"],
                "temperature": round(temp, 1),
                "humidity": humidity,
                "precipitation": round(precip, 2),
                "precipitation_category": p_info["category"],
                "precipitation_level": p_info["level"],
                "precipitation_color": p_info["color"],
                "weather_code": wcode,
                "weather_text": w_text,
                "weather_icon": w_icon,
                "wind_speed_kmh": round(wspeed, 1),
                "wind_direction_deg": wdir,
                "wind_gusts_kmh": round(wgusts, 1),
                "u": round(u_comp, 2),
                "v": round(v_comp, 2)
            }
            grid_output.append(point_data)
            wind_vectors.append({
                "lat": grid_meta["lat"],
                "lng": grid_meta["lng"],
                "speed": round(wspeed, 1),
                "deg": wdir,
                "u": round(u_comp, 2),
                "v": round(v_comp, 2)
            })

        response_data = {
            "status": "success",
            "source": "Open-Meteo Free Atmospheric API",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "grid_points": grid_output,
            "wind_vectors": wind_vectors,
            "max_precipitation_mm": round(max_precip, 2)
        }

        WEATHER_CACHE["grid"] = {
            "timestamp": now,
            "data": response_data
        }
        return response_data

    except Exception as err:
        print(f"[Open-Meteo] Weather API fetch error: {err}. Using fallback telemetry.")
        # Return fallback if network fails
        return _fallback_weather_grid()


def fetch_hydrology_data() -> Dict[str, Any]:
    """
    Fetch GloFAS river discharge (m³/s) for 5 major Bandung river basins from Open-Meteo Flood API.
    """
    now = time.time()
    cached = HYDRO_CACHE.get("rivers")
    if cached and (now - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]

    lats = [str(st["lat"]) for st in BANDUNG_RIVER_STATIONS]
    lngs = [str(st["lng"]) for st in BANDUNG_RIVER_STATIONS]

    url = (
        f"https://flood-api.open-meteo.com/v1/flood?"
        f"latitude={','.join(lats)}&longitude={','.join(lngs)}"
        f"&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max"
        f"&forecast_days=3"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BandungCCTVPlatform/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_data = json.loads(resp.read().decode("utf-8"))

        results_list = raw_data if isinstance(raw_data, list) else [raw_data]

        stations_output = []
        for idx, station in enumerate(BANDUNG_RIVER_STATIONS):
            daily = {}
            if idx < len(results_list):
                daily = results_list[idx].get("daily", {})

            discharges = daily.get("river_discharge", []) or [station["normal_discharge"]]
            max_discharges = daily.get("river_discharge_max", []) or discharges

            cur_discharge = discharges[0] if discharges and discharges[0] is not None else station["normal_discharge"]
            peak_24h = max_discharges[0] if max_discharges and max_discharges[0] is not None else cur_discharge
            peak_72h = max(d for d in max_discharges if d is not None) if any(d is not None for d in max_discharges) else peak_24h

            # Evaluate Flood Risk Stage
            if cur_discharge >= station["danger_threshold"] or peak_24h >= station["danger_threshold"]:
                status = "SIAGA (BAHAYA)"
                status_class = "siaga"
                status_color = "#EF4444"
                status_icon = "🚨"
            elif cur_discharge >= station["warning_threshold"] or peak_24h >= station["warning_threshold"]:
                status = "WASPADA"
                status_class = "waspada"
                status_color = "#F59E0B"
                status_icon = "⚠️"
            else:
                status = "NORMAL"
                status_class = "normal"
                status_color = "#10B981"
                status_icon = "✅"

            stations_output.append({
                "id": station["id"],
                "name": station["name"],
                "river": station["river"],
                "lat": station["lat"],
                "lng": station["lng"],
                "drainage_area": station["drainage_area"],
                "description": station["description"],
                "normal_discharge": station["normal_discharge"],
                "warning_threshold": station["warning_threshold"],
                "danger_threshold": station["danger_threshold"],
                "current_discharge_m3s": round(float(cur_discharge), 2),
                "peak_24h_m3s": round(float(peak_24h), 2),
                "peak_72h_m3s": round(float(peak_72h), 2),
                "status": status,
                "status_class": status_class,
                "status_color": status_color,
                "status_icon": status_icon
            })

        response_data = {
            "status": "success",
            "source": "Open-Meteo GloFAS River Flood API",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "stations": stations_output
        }

        HYDRO_CACHE["rivers"] = {
            "timestamp": now,
            "data": response_data
        }
        return response_data

    except Exception as err:
        print(f"[Open-Meteo] Flood API fetch error: {err}. Using fallback hydrology.")
        return _fallback_hydrology()


def _fallback_weather_grid() -> Dict[str, Any]:
    """Realistic synthetic fallback grid data for Bandung basin."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    grid_output = []
    wind_vectors = []
    for pt in BANDUNG_WEATHER_GRID:
        temp = 28.5 if "Pusat" in pt["zone"] or "Timur" in pt["zone"] else 22.0
        precip = 0.5 if "Utara" in pt["zone"] else 0.0
        p_info = get_precipitation_category(precip)
        w_text, w_icon, w_color = decode_wmo_weather(2 if precip == 0 else 61)
        wspeed = 10.5
        wdir = 230
        rad = math.radians(wdir)
        speed_ms = wspeed / 3.6
        u_comp = -speed_ms * math.sin(rad)
        v_comp = -speed_ms * math.cos(rad)

        grid_output.append({
            "id": pt["id"],
            "name": pt["name"],
            "zone": pt["zone"],
            "lat": pt["lat"],
            "lng": pt["lng"],
            "temperature": temp,
            "humidity": 68,
            "precipitation": precip,
            "precipitation_category": p_info["category"],
            "precipitation_level": p_info["level"],
            "precipitation_color": p_info["color"],
            "weather_code": 2,
            "weather_text": w_text,
            "weather_icon": w_icon,
            "wind_speed_kmh": wspeed,
            "wind_direction_deg": wdir,
            "wind_gusts_kmh": 18.0,
            "u": round(u_comp, 2),
            "v": round(v_comp, 2)
        })
        wind_vectors.append({
            "lat": pt["lat"],
            "lng": pt["lng"],
            "speed": wspeed,
            "deg": wdir,
            "u": round(u_comp, 2),
            "v": round(v_comp, 2)
        })

    return {
        "status": "success",
        "source": "Open-Meteo (Local Fallback)",
        "timestamp": now,
        "grid_points": grid_output,
        "wind_vectors": wind_vectors,
        "max_precipitation_mm": 0.5
    }


def _fallback_hydrology() -> Dict[str, Any]:
    """Realistic fallback hydrology data for Bandung rivers."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    stations_output = []
    for st in BANDUNG_RIVER_STATIONS:
        stations_output.append({
            "id": st["id"],
            "name": st["name"],
            "river": st["river"],
            "lat": st["lat"],
            "lng": st["lng"],
            "drainage_area": st["drainage_area"],
            "description": st["description"],
            "normal_discharge": st["normal_discharge"],
            "warning_threshold": st["warning_threshold"],
            "danger_threshold": st["danger_threshold"],
            "current_discharge_m3s": st["normal_discharge"],
            "peak_24h_m3s": st["normal_discharge"] * 1.2,
            "peak_72h_m3s": st["normal_discharge"] * 1.4,
            "status": "NORMAL",
            "status_class": "normal",
            "status_color": "#10B981",
            "status_icon": "✅"
        })

    return {
        "status": "success",
        "source": "Open-Meteo GloFAS (Local Fallback)",
        "timestamp": now,
        "stations": stations_output
    }
