import os
import gzip
import json
import socket
import threading
import time
import urllib.request
import urllib.parse
import urllib.error
import csv
import requests
try:
    import cv2
except ImportError:
    cv2 = None
from requests.adapters import HTTPAdapter
from urllib3.util import parse_url
from urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
import concurrent.futures
from collections import OrderedDict
from flask import Flask, render_template, jsonify, request, Response
from script_tools.update_cctv_links import sync_catalog_files

app = Flask(__name__)

COMPRESSIBLE_TYPES = {"text/html", "application/json"}


@app.after_request
def gzip_response(response):
    """Gzip large HTML/JSON (the page embeds every marker). Streams and video are left untouched."""
    if (
        response.status_code != 200
        or response.mimetype not in COMPRESSIBLE_TYPES
        or response.direct_passthrough
        or response.is_streamed
        or "Content-Encoding" in response.headers
        or "gzip" not in request.accept_encodings
    ):
        return response
    body = response.get_data()
    if len(body) < 1024:
        return response
    response.set_data(gzip.compress(body, compresslevel=5))
    response.headers["Content-Encoding"] = "gzip"
    response.vary.add("Accept-Encoding")
    return response


# ==========================================
# Paths & Data Files
# ==========================================
CSV_PATH = Path("csv/cctv_bandung_data_validated.csv")
CSV_RTSP_PATH = Path("csv/cctv_rtsp_telkom.csv")
RTSP_URL_BY_ID = {}


def _load_dotenv_if_present(dotenv_path: Path = Path(".env")) -> None:
    """Populate os.environ from a local .env file without overriding existing env vars."""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv_if_present()
RTSP_USER = os.environ.get("RTSP_USER", "").strip()
RTSP_PASS = os.environ.get("RTSP_PASS", "").strip()
CSV_PEREMPATAN_DETAIL_PATH = Path("csv/cctv_perempatan_detail.csv")
CSV_PERSIMPANGAN_DETAIL_PATH = Path("csv/cctv_semua_persimpangan_detail.csv")
CSV_CLUSTERS_PATH = Path("csv/cctv_perempatan_clusters.csv")
CSV_PERSPECTIVES_PATH = Path("csv/cctv_camera_perspectives.csv")
ROI_JSON_PATH = Path("outputs/roi_annotations.json")
TELEMETRY_JSON_PATH = Path("outputs/traffic_telemetry.json")

UPSTREAM_VIDEO_BASE = "https://pelindung.bandung.go.id:3443/video/"
GLOBAL_CCTV_STATUS = {}
CCTV_FAIL_STREAK = {}
markers_global = []
intersections_global = []
CATALOG_SYNC_INTERVAL_SECONDS = 900
CATALOG_LOCK = threading.Lock()
CATALOG_STATE = {
    "version": 1,
    "count": 0,
    "last_synced_at": None,
    "last_error": None,
    "added": 0,
    "updated": 0,
    "removed": 0,
}

UPSTREAM_CONNECT_TIMEOUT = 6.0
UPSTREAM_READ_TIMEOUT = 25.0
UPSTREAM_RETRY = Retry(
    total=3,
    connect=3,
    read=2,
    backoff_factor=0.4,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "HEAD"}),
    raise_on_status=False,
)
UPSTREAM_SESSION = requests.Session()
_upstream_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=64, max_retries=UPSTREAM_RETRY)
UPSTREAM_SESSION.mount("https://", _upstream_adapter)
UPSTREAM_SESSION.mount("http://", _upstream_adapter)
UPSTREAM_SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
)

# Separate, retry-free session for cheap liveness probing: a health check must be fast,
# not certain — transient failures are absorbed by CCTV_FAIL_STREAK across scan cycles,
# so retrying here only turns one flaky camera into a multi-second stall for every scan.
HEALTH_SESSION = requests.Session()
_health_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=64, max_retries=0)
HEALTH_SESSION.mount("https://", _health_adapter)
HEALTH_SESSION.mount("http://", _health_adapter)
HEALTH_SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

# Stream Cache for .ts video chunks (in-memory TTL cache to accelerate Multi-View Grid)
STREAM_CACHE_LOCK = threading.Lock()
STREAM_CACHE = OrderedDict()
MAX_CACHE_ITEMS = 300
CACHE_TTL_SECONDS = 8.0

REFRESH_LOCK = threading.Lock()
REFRESH_STATE = {
    "is_refreshing": False,
    "total": 0,
    "completed": 0,
    "active_count": 0,
    "missing_count": 0,
    "last_refreshed_at": None,
    "duration_seconds": 0.0
}

# ==========================================
# Major Traffic Corridors Definition
# ==========================================
CORRIDORS_CONFIG = [
    {
        "id": "koridor_asia_afrika",
        "name": "Koridor 1: Asia Afrika - Sudirman - Banceuy",
        "category": "Pusat Kota & Heritage",
        "description": "Jantung komersial & wisata Kota Bandung (Alun-Alun, Banceuy, Braga, Sudirman)",
        "keywords": ["alun_alun", "asia_afrika", "banceuy", "braga", "sudirman", "naripan", "tamblong"]
    },
    {
        "id": "koridor_dago_merdeka",
        "name": "Koridor 2: Ir. H. Juanda (Dago) - Merdeka - BIP",
        "category": "Utara - Selatan",
        "description": "Pusat belanja, kantor pemerintahan, dan wisata mode Dago",
        "keywords": ["dago", "juanda", "merdeka", "bip", "riau", "balai_kota", "wastukencana"]
    },
    {
        "id": "koridor_pasteur_surapati",
        "name": "Koridor 3: Dr. Djunjunan (Pasteur) - Pasirkaliki - Pasupati",
        "category": "Gerbang Tol & Flyover",
        "description": "Akses utama pintu masuk Tol Pasteur menuju jembatan layang Prof. Mochtar Kusumaatmadja",
        "keywords": ["pasteur", "djunjunan", "pasirkaliki", "surapati", "pasupati", "gasibu", "sukajadi"]
    },
    {
        "id": "koridor_soekarno_hatta",
        "name": "Koridor 4: Soekarno-Hatta (Bypass Arteri)",
        "category": "Arteri Primer",
        "description": "Jalur lingkar arteri penghubung Bandung Barat, Selatan, dan Timur",
        "keywords": ["soekarno_hatta", "bypass", "kopo", "moch_toha", "buah_batu", "kircon", "gedebage", "cibiru"]
    },
    {
        "id": "koridor_buah_batu",
        "name": "Koridor 5: Buah Batu - Kordon - Bojongsoang",
        "category": "Kawasan Selatan",
        "description": "Akses utama pintu tol Buah Batu dan perbatasan Kabupaten Bandung",
        "keywords": ["buah_batu", "bubat", "kordon", "griya", "banteng", "turangga", "bojongsoang"]
    },
    {
        "id": "koridor_cihampelas_setiabudi",
        "name": "Koridor 6: Cihampelas - Cipaganti - Setiabudi",
        "category": "Wisata & Kampus",
        "description": "Kawasan wisata Skywalk Cihampelas dan jalur utama menuju Lembang",
        "keywords": ["cihampelas", "cipaganti", "setiabudi", "gandok", "skywalk", "ledeng", "sukajadi"]
    },
    {
        "id": "koridor_riau_laswi",
        "name": "Koridor 7: R.E. Martadinata (Riau) - Laswi - Gatot Subroto",
        "category": "Kuliner & Lifestyle",
        "description": "Kawasan factory outlet, Trans Studio, dan simpang Laswi",
        "keywords": ["riau", "martadinata", "laswi", "gatot_subroto", "tsm", "patrakomala", "supratman"]
    }
]

# ==========================================
# Flood-Prone Hotspots (Early Warning Matrix)
# ==========================================
FLOOD_HOTSPOTS_CONFIG = [
    {
        "id": "flood_pasteur",
        "name": "Kawasan Pasteur (Dr. Djunjunan / BTC)",
        "lat": -6.8931,
        "lng": 107.5852,
        "drainage": "Sungai Citepus",
        "cctv_keywords": ["pasteur", "djunjunan", "btc", "sukagalih"],
        "critical_threshold_mm": 20.0
    },
    {
        "id": "flood_gedebage",
        "name": "Kawasan Pasar Gedebage / Rumah Sakit",
        "lat": -6.9442,
        "lng": 107.6894,
        "drainage": "Sungai Cinambo",
        "cctv_keywords": ["gedebage", "cinambo", "derwati", "pasar_induk"],
        "critical_threshold_mm": 18.0
    },
    {
        "id": "flood_pagarsih",
        "name": "Jl. Pagarsih / Astana Anyar",
        "lat": -6.9248,
        "lng": 107.5975,
        "drainage": "Sungai Citepus Hilir",
        "cctv_keywords": ["pagarsih", "astana_anyar", "sudirman", "kelenteng"],
        "critical_threshold_mm": 15.0
    },
    {
        "id": "flood_kopo_citarip",
        "name": "Simpang Kopo - Citarip / Leuwi Panjang",
        "lat": -6.9389,
        "lng": 107.5921,
        "drainage": "Saluran Pematusan Kopo",
        "cctv_keywords": ["kopo", "citarip", "leuwi_panjang", "soekarno_hatta_kopo"],
        "critical_threshold_mm": 22.0
    },
    {
        "id": "flood_cingised",
        "name": "Kawasan Cingised / Arcamanik",
        "lat": -6.9212,
        "lng": 107.6743,
        "drainage": "Sungai Cironggeng",
        "cctv_keywords": ["cingised", "cisaranten", "arcamanik", "sukamiskin"],
        "critical_threshold_mm": 25.0
    }
]


# ==========================================
# Data Parsers & Loaders
# ==========================================
def parse_location(value: str):
    parts = [part.strip() for part in (value or "").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lng = float(parts[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return lat, lng


def load_intersection_clusters():
    """Load road intersection clusters where CCTVs are located."""
    clusters = []
    if not CSV_CLUSTERS_PATH.exists():
        return clusters
    with CSV_CLUSTERS_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lat = float(row.get("Latitude", 0))
                lng = float(row.get("Longitude", 0))
            except ValueError:
                continue
            cctv_names = [n.strip() for n in row.get("CCTV_Names", "").split("|") if n.strip()]
            clusters.append({
                "id": row.get("Intersection_ID", "").strip(),
                "name": row.get("Intersection_Name", "").strip(),
                "type": row.get("Intersection_Type", "Perempatan (4-way)").strip(),
                "roads": row.get("Road_Names", "").strip(),
                "lat": lat,
                "lng": lng,
                "hasTrafficSignals": row.get("Has_Traffic_Signals", "NO").strip() == "YES",
                "cctvCount": int(row.get("CCTV_Count", 0) or len(cctv_names)),
                "cctvNames": cctv_names,
                "cctvProviders": row.get("CCTV_Providers", "").strip(),
                "avgDistance": round(float(row.get("Avg_Distance_Meters", 0) or 0), 1),
                "minDistance": round(float(row.get("Min_Distance_Meters", 0) or 0), 1),
                "maxDistance": round(float(row.get("Max_Distance_Meters", 0) or 0), 1),
                "osmNodes": row.get("OSM_Node_IDs", "").strip()
            })
    return clusters


def load_roi_annotations() -> dict:
    """Load ROI polygon ground truth annotations."""
    if not ROI_JSON_PATH.exists():
        return {}
    try:
        with open(ROI_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {e["cctv_id"]: e for e in data if isinstance(e, dict) and "cctv_id" in e}
    except Exception:
        return {}
    return {}


BUAHBATU_FORECAST_PATH = Path("outputs/buahbatu_forecast.json")
_BUAHBATU_CACHE = {"mtime": None, "data": None}
_BUAHBATU_LOCK = threading.Lock()


def load_buahbatu_forecast() -> dict | None:
    """Replay forecast for CCTV Terusan Buah Batu - Jembatan Tol 02 (extractions/forecast/ekspor_app.py)."""
    if not BUAHBATU_FORECAST_PATH.exists():
        return None
    mtime = BUAHBATU_FORECAST_PATH.stat().st_mtime
    with _BUAHBATU_LOCK:
        if _BUAHBATU_CACHE["mtime"] != mtime:
            with open(BUAHBATU_FORECAST_PATH, "r", encoding="utf-8") as f:
                _BUAHBATU_CACHE["data"] = json.load(f)
            _BUAHBATU_CACHE["mtime"] = mtime
        return _BUAHBATU_CACHE["data"]


def load_traffic_telemetry() -> dict:
    """Load cached speed estimation and Level of Service telemetry."""
    if not TELEMETRY_JSON_PATH.exists():
        return {}
    try:
        with open(TELEMETRY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_markers(rows: list[dict] | None = None):
    markers = []
    if rows is None:
        if not CSV_PATH.exists():
            return markers
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
            row_iterable = list(csv.DictReader(handle))
    else:
        row_iterable = list(rows)

    if CSV_RTSP_PATH.exists():
        with CSV_RTSP_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
            row_iterable.extend(csv.DictReader(handle))
    # Load intersection lookup dictionaries
    persimpangan_map = {}
    if CSV_PERSIMPANGAN_DETAIL_PATH.exists():
        with CSV_PERSIMPANGAN_DETAIL_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                persimpangan_map[r.get("CCTV_Name", "").strip()] = r

    perempatan_map = {}
    if CSV_PEREMPATAN_DETAIL_PATH.exists():
        with CSV_PEREMPATAN_DETAIL_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                perempatan_map[r.get("CCTV_Name", "").strip()] = r

    perspectives_map = {}
    if CSV_PERSPECTIVES_PATH.exists():
        with CSV_PERSPECTIVES_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                perspectives_map[r.get("cctv_name", "").strip()] = r

    roi_data = load_roi_annotations()
    telemetry_data = load_traffic_telemetry()

    for row in row_iterable:
            point = parse_location(row.get("Location", ""))
            if point is None:
                continue
            lat, lng = point
            name = row.get("CCTV_Name", "").strip()
            sample_file_name = os.path.basename(row.get("Sample_File", "").strip())
            cctv_id = os.path.splitext(sample_file_name or name.lower().replace(" ", "_"))[0]

            raw_stream = row.get("m3u8_URL", "").strip()
            is_rtsp = raw_stream.lower().startswith("rtsp://")
            stream_type = "rtsp" if is_rtsp else "hls"

            if is_rtsp:
                parsed_rtsp = urllib.parse.urlparse(raw_stream)
                host_port = parsed_rtsp.hostname or ""
                if parsed_rtsp.port:
                    host_port = f"{host_port}:{parsed_rtsp.port}"
                if parsed_rtsp.username:
                    RTSP_URL_BY_ID[cctv_id] = raw_stream
                elif RTSP_USER and RTSP_PASS:
                    auth = f"{urllib.parse.quote(RTSP_USER, safe='')}:{urllib.parse.quote(RTSP_PASS, safe='')}@"
                    RTSP_URL_BY_ID[cctv_id] = f"rtsp://{auth}{host_port}{parsed_rtsp.path}"
                else:
                    RTSP_URL_BY_ID[cctv_id] = raw_stream
                normalized_stream = f"rtsp://{host_port}{parsed_rtsp.path}"
                proxy_stream = f"/api/rtsp_mjpeg?id={urllib.parse.quote(cctv_id, safe='')}"
            else:
                # Normalize pelindung URL with port 3443
                if raw_stream.startswith("https://pelindung.bandung.go.id/video/"):
                    normalized_stream = raw_stream.replace(
                        "https://pelindung.bandung.go.id/video/",
                        "https://pelindung.bandung.go.id:3443/video/"
                    )
                else:
                    normalized_stream = raw_stream

                # Compute internal proxy URL
                if "/video/" in normalized_stream:
                    subpath = normalized_stream.split("/video/", 1)[1]
                    proxy_stream = f"/api/stream/{subpath}"
                elif normalized_stream:
                    proxy_stream = f"/api/proxy_stream?url={urllib.parse.quote(normalized_stream, safe='')}"
                else:
                    proxy_stream = ""

            # Intersection associations
            isect_all = persimpangan_map.get(name, {})
            isect_perempatan = perempatan_map.get(name, {})

            isect_lat = None
            isect_lng = None
            if isect_all.get("Intersection_Latitude") and isect_all.get("Intersection_Longitude"):
                try:
                    isect_lat = float(isect_all["Intersection_Latitude"])
                    isect_lng = float(isect_all["Intersection_Longitude"])
                except ValueError:
                    pass

            dist_all = 0.0
            try:
                dist_all = float(isect_all.get("Distance_to_Intersection_Meters", 0) or 0)
            except ValueError:
                pass

            dist_perempatan = 0.0
            try:
                dist_perempatan = float(isect_perempatan.get("Distance_to_Perempatan_Meters", 0) or 0)
            except ValueError:
                pass

            # Perspective & Optical Flow Attributes
            persp = perspectives_map.get(name, {})
            bearing_val = 0.0
            try:
                bearing_val = float(persp.get("bearing_deg", 0.0) or 0.0)
            except ValueError:
                pass
            conf_val = 0.7
            try:
                conf_val = float(persp.get("confidence_score", 0.7) or 0.7)
            except ValueError:
                pass

            # ROI & Traffic Telemetry Enriched Data
            roi_entry = roi_data.get(cctv_id) or roi_data.get(name) or {}
            has_rois = len(roi_entry.get("rois", [])) > 0
            rois_list = roi_entry.get("rois", [])

            t_entry = telemetry_data.get(cctv_id) or telemetry_data.get(name) or {}
            avg_speed = t_entry.get("avg_speed_kmh", None)
            los_info = t_entry.get("level_of_service", {
                "code": "B" if has_rois else "—",
                "description": "Lancar" if has_rois else "Belum Dianalisis",
                "color": "#3B82F6" if has_rois else "#94A3B8",
                "congestion_index": 0.25 if has_rois else 0.0
            })

            markers.append({
                "cctvId": cctv_id,
                "name": name,
                "lat": lat,
                "lng": lng,
                "provider": row.get("Provider", "").strip(),
                "stream": normalized_stream,
                "proxyStream": proxy_stream,
                "isRtsp": is_rtsp,
                "streamType": stream_type,
                "sampleDownloaded": row.get("Sample_Downloaded", "").strip(),
                "sampleFile": row.get("Sample_File", "").strip(),
                "sampleError": row.get("Sample_Error", "").strip(),
                # Intersection Data
                "intersectionId": isect_all.get("Nearest_Intersection_ID", ""),
                "intersectionName": isect_all.get("Nearest_Intersection_Name", ""),
                "intersectionType": isect_all.get("Intersection_Type", "Persimpangan"),
                "intersectingRoads": isect_all.get("Intersecting_Roads", ""),
                "intersectionLat": isect_lat,
                "intersectionLng": isect_lng,
                "intersectionDistance": round(dist_all, 1),
                "within100m": isect_all.get("Within_Radius_100m", "NO") == "YES",
                "perempatanName": isect_perempatan.get("Nearest_Perempatan_Name", ""),
                "perempatanDistance": round(dist_perempatan, 1),
                "within100mPerempatan": isect_perempatan.get("Within_Radius_100m", "NO") == "YES",
                "hasTrafficSignals": isect_all.get("Has_Traffic_Signals", "NO") == "YES" or isect_perempatan.get("Has_Traffic_Signals", "NO") == "YES",
                # Perspective Data
                "bearing": round(bearing_val, 1),
                "cardinal": persp.get("cardinal_direction", "Utara"),
                "fov": float(persp.get("fov_deg", 60.0) or 60.0),
                "perspectiveMethod": persp.get("estimation_method", "OSM_GEOMETRIC_ALIGNMENT"),
                "confidence": round(conf_val, 2),
                "trafficOrientation": persp.get("traffic_orientation", "NOT_ANALYZED"),
                "opticalFlowDetected": persp.get("optical_flow_detected", "False") == "True",
                # ROI & Telemetry Data
                "hasRois": has_rois,
                "roiCount": len(rois_list),
                "rois": rois_list,
                "avgSpeedKmh": avg_speed,
                "levelOfService": los_info,
                "flowPerHour": t_entry.get("flow_per_hour", 0),
                "totalVehicles": t_entry.get("total_vehicles", 0)
            })
    return markers


# ==========================================
# Asynchronous Camera Health Checker
# ==========================================
def check_single_cctv(marker_item):
    """Probes a single CCTV playlist (HTTP HEAD) or RTSP endpoint (TCP/OPTIONS)."""
    name = marker_item.get("name")
    url = marker_item.get("stream")
    if not url:
        return name, "NO"
    if marker_item.get("isRtsp") or url.lower().startswith("rtsp://"):
        rtsp_url = RTSP_URL_BY_ID.get(marker_item.get("cctvId", ""), url)
        parsed = urllib.parse.urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return name, "NO"
        try:
            with socket.create_connection((host, port), timeout=3.5) as sock:
                sock.settimeout(3.5)
                req = f"OPTIONS {rtsp_url} RTSP/1.0\r\nCSeq: 1\r\n\r\n".encode("ascii", errors="ignore")
                sock.sendall(req)
                resp = sock.recv(256).decode("ascii", errors="ignore")
                return name, "YES" if "RTSP/1.0" in resp else "NO"
        except OSError:
            return name, "NO"
    try:
        res = HEALTH_SESSION.head(url, timeout=(3.5, 5.0), allow_redirects=True)
        return name, "YES" if res.status_code == 200 else "NO"
    except requests.RequestException:
        return name, "NO"


def execute_full_refresh():
    """Concurrently checks live status for all CCTV cameras using ThreadPoolExecutor."""
    global GLOBAL_CCTV_STATUS, markers_global, REFRESH_STATE

    markers = markers_global
    if not markers:
        return REFRESH_STATE

    with REFRESH_LOCK:
        if REFRESH_STATE["is_refreshing"]:
            return REFRESH_STATE
        REFRESH_STATE["is_refreshing"] = True
        REFRESH_STATE["total"] = len(markers)
        REFRESH_STATE["completed"] = 0

    t0 = time.time()
    results = {}
    active_cnt = 0
    missing_cnt = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            future_to_cam = {executor.submit(check_single_cctv, m): m for m in markers}
            for future in concurrent.futures.as_completed(future_to_cam):
                try:
                    name, probe = future.result()
                    if probe == "YES":
                        CCTV_FAIL_STREAK[name] = 0
                        status = "YES"
                    else:
                        CCTV_FAIL_STREAK[name] = CCTV_FAIL_STREAK.get(name, 0) + 1
                        status = "NO" if CCTV_FAIL_STREAK[name] >= 2 else GLOBAL_CCTV_STATUS.get(name, "NO")
                    results[name] = status
                    if status == "YES":
                        active_cnt += 1
                    else:
                        missing_cnt += 1
                    with REFRESH_LOCK:
                        REFRESH_STATE["completed"] = len(results)
                except Exception:
                    pass

        # Update in-memory statuses
        for m in markers_global:
            name = m.get("name")
            if name in results:
                m["sampleDownloaded"] = results[name]
                GLOBAL_CCTV_STATUS[name] = results[name]

        duration = round(time.time() - t0, 2)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with REFRESH_LOCK:
            REFRESH_STATE["is_refreshing"] = False
            REFRESH_STATE["completed"] = len(results)
            REFRESH_STATE["active_count"] = active_cnt
            REFRESH_STATE["missing_count"] = missing_cnt
            REFRESH_STATE["last_refreshed_at"] = now_str
            REFRESH_STATE["duration_seconds"] = duration

    except Exception as e:
        with REFRESH_LOCK:
            REFRESH_STATE["is_refreshing"] = False
        print(f"Error during CCTV refresh: {e}")

    return REFRESH_STATE

def apply_catalog_rows(rows: list[dict]) -> None:
    """Rebuild global markers from new catalog rows and update catalog state."""
    global markers_global, GLOBAL_CCTV_STATUS, CCTV_FAIL_STREAK, CATALOG_STATE
    new_markers = load_markers(rows)
    new_names = set()
    for m in new_markers:
        name = m.get("name")
        if not name:
            continue
        new_names.add(name)
        if name in GLOBAL_CCTV_STATUS:
            m["sampleDownloaded"] = GLOBAL_CCTV_STATUS[name]
        else:
            GLOBAL_CCTV_STATUS[name] = m.get("sampleDownloaded", "NO")

    stale_names = [name for name in list(GLOBAL_CCTV_STATUS.keys()) if name not in new_names]
    for name in stale_names:
        GLOBAL_CCTV_STATUS.pop(name, None)
        CCTV_FAIL_STREAK.pop(name, None)

    with CATALOG_LOCK:
        markers_global = new_markers
        CATALOG_STATE["version"] += 1
        CATALOG_STATE["count"] = len(new_markers)


def sync_cctv_catalog() -> bool:
    """Sync catalog with Pelindung Bandung API. Returns True if catalog changed."""
    try:
        rows, stats = sync_catalog_files()
    except Exception as exc:
        with CATALOG_LOCK:
            CATALOG_STATE["last_error"] = str(exc)
        print(f"[catalog] Sinkronisasi Pelindung gagal: {exc}")
        return False

    changed = bool(stats.get("changed", False))
    if changed:
        apply_catalog_rows(rows)

    with CATALOG_LOCK:
        CATALOG_STATE["last_synced_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        CATALOG_STATE["last_error"] = stats.get("write_error")
        CATALOG_STATE["added"] = stats.get("added", 0)
        CATALOG_STATE["updated"] = stats.get("updated", 0)
        CATALOG_STATE["removed"] = stats.get("removed", 0)

    print(
        f"[catalog] {len(rows)} CCTV dari Pelindung: +{stats.get('added', 0)} baru, "
        f"{stats.get('updated', 0)} diperbarui, -{stats.get('removed', 0)} dihapus"
    )
    return changed


def catalog_sync_worker():
    """Background worker syncing catalog at startup and periodically."""
    while True:
        try:
            if sync_cctv_catalog():
                execute_full_refresh()
        except Exception as exc:
            print(f"[catalog] Background sync worker error: {exc}")
        time.sleep(CATALOG_SYNC_INTERVAL_SECONDS)


def background_periodic_worker():
    """Background worker that periodically refreshes CCTV statuses every 5 minutes."""
    time.sleep(2)
    while True:
        execute_full_refresh()
        time.sleep(300)


# Initialize global state
markers_global = load_markers()
intersections_global = load_intersection_clusters()

for m in markers_global:
    GLOBAL_CCTV_STATUS[m['name']] = m.get('sampleDownloaded', 'NO')

CATALOG_STATE["count"] = len(markers_global)

catalog_thread = threading.Thread(target=catalog_sync_worker, daemon=True)
catalog_thread.start()
# Start background periodic refresh thread
t = threading.Thread(target=background_periodic_worker, daemon=True)
t.start()


# ==========================================
# Web Routes
# ==========================================
@app.route("/")
def index():
    with CATALOG_LOCK:
        current_markers = list(markers_global)
        catalog_ver = CATALOG_STATE["version"]
    marker_json = json.dumps(current_markers, ensure_ascii=False)
    intersections_json = json.dumps(intersections_global, ensure_ascii=False)
    owm_api_key = os.environ.get("OWM_API_KEY", "")
    return render_template(
        "index.html",
        marker_json=marker_json,
        marker_count=len(current_markers),
        intersections_json=intersections_json,
        intersections_count=len(intersections_global),
        owm_api_key=owm_api_key,
        catalog_version=catalog_ver,
    )


@app.route("/api/intersections")
def api_intersections():
    """Get list of all 132 road intersection clusters with monitoring CCTVs."""
    return jsonify(intersections_global)


@app.route("/api/perspectives")
def api_perspectives():
    """Get list of all CCTV camera headings, bearings, FOVs, and optical flow perspectives."""
    perspectives = [
        {
            "name": m["name"],
            "lat": m["lat"],
            "lng": m["lng"],
            "bearing": m.get("bearing", 0.0),
            "cardinal": m.get("cardinal", "Utara"),
            "fov": m.get("fov", 60.0),
            "method": m.get("perspectiveMethod", "OSM_GEOMETRIC_ALIGNMENT"),
            "confidence": m.get("confidence", 0.7),
            "trafficOrientation": m.get("trafficOrientation", "NOT_ANALYZED"),
            "opticalFlowDetected": m.get("opticalFlowDetected", False)
        }
        for m in markers_global
    ]
    return jsonify(perspectives)


@app.route("/api/rois")
def api_rois():
    """Get all ROI annotations mapped by CCTV ID."""
    data = load_roi_annotations()
    return jsonify(data)


@app.route("/api/traffic-telemetry")
def api_traffic_telemetry():
    """Get vehicle speed estimates, volume flow, and Level of Service (LOS) across all CCTVs."""
    telemetry = load_traffic_telemetry()
    return jsonify({
        "status": "success",
        "count": len(telemetry),
        "telemetry": telemetry
    })


@app.route("/api/corridors")
def api_corridors():
    """Get list of road corridors and their associated active CCTVs for Multi-View Grid."""
    corridors_result = []
    for c in CORRIDORS_CONFIG:
        c_kws = c["keywords"]
        matched_cctvs = []
        for m in markers_global:
            m_text = (m["name"] + " " + m.get("sampleFile", "")).lower()
            if any(kw in m_text for kw in c_kws):
                matched_cctvs.append(m)

        corridors_result.append({
            "id": c["id"],
            "name": c["name"],
            "category": c["category"],
            "description": c["description"],
            "cameraCount": len(matched_cctvs),
            "cameras": matched_cctvs
        })
    return jsonify(corridors_result)


@app.route("/api/flood-risk")
def api_flood_risk():
    """Get real-time flood-prone monitoring hotspots and correlated CCTV camera status."""
    from script_tools.open_meteo_service import fetch_weather_grid_data, fetch_hydrology_data
    try:
        w_data = fetch_weather_grid_data()
        max_rain = w_data.get("max_precipitation_mm", 0.0)
    except Exception:
        max_rain = 0.0

    try:
        h_data = fetch_hydrology_data()
        river_stations = h_data.get("stations", [])
    except Exception:
        river_stations = []

    hotspots = []
    for h in FLOOD_HOTSPOTS_CONFIG:
        kws = h["cctv_keywords"]
        matched = []
        for m in markers_global:
            m_text = (m["name"] + " " + m.get("sampleFile", "")).lower()
            if any(kw in m_text for kw in kws):
                matched.append({
                    "name": m["name"],
                    "cctvId": m.get("cctvId"),
                    "lat": m["lat"],
                    "lng": m["lng"],
                    "status": m.get("sampleDownloaded", "NO"),
                    "proxyStream": m.get("proxyStream", "")
                })

        # Enrich with live Open-Meteo hydrology & rainfall if available
        hydro_match = None
        for r in river_stations:
            if h["name"].lower().split()[0] in r["name"].lower() or h["drainage"].lower() in r["name"].lower():
                hydro_match = r
                break

        current_rain = max_rain
        risk_level = "NORMAL"
        risk_color = "#10B981"
        if hydro_match:
            if hydro_match["status_class"] == "siaga" or current_rain >= h["critical_threshold_mm"]:
                risk_level = "SIAGA (BAHAYA)"
                risk_color = "#EF4444"
            elif hydro_match["status_class"] == "waspada" or current_rain >= (h["critical_threshold_mm"] * 0.4):
                risk_level = "WASPADA"
                risk_color = "#F59E0B"

        hotspots.append({
            "id": h["id"],
            "name": h["name"],
            "lat": h["lat"],
            "lng": h["lng"],
            "drainage": h["drainage"],
            "criticalThresholdMm": h["critical_threshold_mm"],
            "cctvCount": len(matched),
            "cctvs": matched,
            "currentRainMm": round(current_rain, 2),
            "riverDischarge": hydro_match.get("current_discharge_m3s") if hydro_match else None,
            "riverPeak24h": hydro_match.get("peak_24h_m3s") if hydro_match else None,
            "riskLevel": risk_level,
            "riskColor": risk_color
        })
    return jsonify({
        "status": "success",
        "hotspots": hotspots
    })


# ==========================================
# Open-Meteo Free API Integrations
# ==========================================
@app.route("/api/weather/grid")
def api_weather_grid():
    """Real-time Open-Meteo multi-point precipitation & atmospheric grid across Bandung Raya."""
    from script_tools.open_meteo_service import fetch_weather_grid_data
    data = fetch_weather_grid_data()
    return jsonify(data)


@app.route("/api/weather/wind-field")
def api_weather_wind_field():
    """Real-time Open-Meteo wind vector field for animated canvas streamlines."""
    from script_tools.open_meteo_service import fetch_weather_grid_data
    data = fetch_weather_grid_data()
    return jsonify({
        "status": data.get("status", "success"),
        "timestamp": data.get("timestamp"),
        "wind_vectors": data.get("wind_vectors", [])
    })


@app.route("/api/weather/hydrology")
def api_weather_hydrology():
    """Real-time Open-Meteo GloFAS river discharge & flood forecast for Bandung river basins."""
    from script_tools.open_meteo_service import fetch_hydrology_data
    data = fetch_hydrology_data()
    return jsonify(data)


# ==========================================
# Terrain Elevation (Copernicus DEM via Open-Meteo)
# ==========================================
@app.route("/api/elevation/grid")
def api_elevation_grid():
    """Static terrain elevation grid (Copernicus DEM GLO-90) powering the elevation heatmap layer."""
    from script_tools.elevation_service import fetch_elevation_grid_data
    data = fetch_elevation_grid_data(force_refresh=request.args.get("refresh") == "1")
    return jsonify(data)

@app.route("/api/weather/temperature-grid")
def api_weather_temperature_grid():
    """Temperature heatmap grid (24 hourly frames). Placeholder until forecast output is wired in."""
    from script_tools.temperature_service import fetch_temperature_grid_data
    return jsonify(fetch_temperature_grid_data())

@app.route("/api/buahbatu/forecast/meta")
def api_buahbatu_forecast_meta():
    """Camera info, available replay origins, capacity, and per-horizon error for the Buah Batu forecast panel."""
    data = load_buahbatu_forecast()
    if data is None:
        return jsonify({"status": "unavailable", "reason": "outputs/buahbatu_forecast.json belum dibuat"}), 404
    meta = {k: v for k, v in data.items() if k not in ("seri", "asal")}
    meta["asal"] = [{"t": a["t"], "model": a["model"]} for a in data["asal"]]
    meta["status"] = "success"
    return jsonify(meta)


@app.route("/api/buahbatu/forecast")
def api_buahbatu_forecast():
    """Forecast from one replay origin: traffic per direction (+5..+360 min) and 5-minute weather.

    Query: asal=<ISO time from /meta> (default: latest origin). Series start 2 hours before the origin.
    """
    data = load_buahbatu_forecast()
    if data is None:
        return jsonify({"status": "unavailable", "reason": "outputs/buahbatu_forecast.json belum dibuat"}), 404
    origins = data["asal"]
    wanted = request.args.get("asal")
    item = next((a for a in origins if a["t"] == wanted), None) if wanted else origins[-1]
    if item is None:
        return jsonify({"status": "error", "reason": f"titik asal tidak ditemukan: {wanted}"}), 400
    seri = data["seri"]
    hist = 24                                  # 2 jam sebelum asal
    lo, hi = item["i"] - hist, item["i"] + data["h_max"] + 1
    window = {k: v[max(lo, 0):hi] for k, v in seri.items() if isinstance(v, list)}
    return jsonify({
        "status": "success", "asal": item["t"], "model": item["model"], "pred": item["pred"],
        "offset_asal": item["i"] - max(lo, 0), "langkah_menit": data["langkah_menit"], "seri": window,
    })


@app.route("/api/weather/classify-frame", methods=["POST"])
def api_weather_classify_frame():
    """Classify a single JPEG CCTV frame (raw request body) as rain / sunny / unknown (low light)."""
    from script_tools.weather_classifier import classify_frame_bytes
    result = classify_frame_bytes(request.get_data(cache=False))
    status_map = {"success": 200, "error": 400, "unavailable": 503}
    code = status_map.get(result.get("status"), 500)
    return jsonify(result), code


@app.route("/api/stream-events")
def api_stream_events():
    """Server-Sent Events (SSE) feed of CCTV health.

    One event per connection, then close: EventSource reconnects after `retry` ms. A connection that
    stays open would hold a gthread worker thread per browser tab and exhaust the server.
    """
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "refresh_state": REFRESH_STATE,
        "active_cctvs": REFRESH_STATE.get("active_count", 0),
        "total_cctvs": len(markers_global)
    }
    return Response(
        f"retry: 5000\ndata: {json.dumps(payload)}\n\n",
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/api/status")
def api_status():
    if request.args.get("meta") == "1":
        with CATALOG_LOCK:
            catalog_meta = dict(CATALOG_STATE)
        return jsonify({
            "statuses": GLOBAL_CCTV_STATUS,
            "state": REFRESH_STATE,
            "catalog": catalog_meta,
        })
    return jsonify(GLOBAL_CCTV_STATUS)


@app.route("/api/markers")
def api_markers():
    with CATALOG_LOCK:
        current_version = CATALOG_STATE["version"]
        current_markers = list(markers_global)
    return jsonify({
        "version": current_version,
        "markers": current_markers,
    })

@app.route("/api/refresh-status", methods=["GET", "POST"])
def api_refresh_status():
    """Trigger on-demand concurrent refresh of all 375 CCTV cameras."""
    wait_sync = request.args.get("sync") == "1" or request.args.get("wait") == "1" or request.is_json

    if REFRESH_STATE["is_refreshing"]:
        return jsonify({
            "success": True,
            "message": "Pemindaian status sedang berjalan",
            "state": REFRESH_STATE,
            "statuses": GLOBAL_CCTV_STATUS
        })

    if wait_sync:
        state = execute_full_refresh()
        return jsonify({
            "success": True,
            "message": f"Status {state['total']} CCTV berhasil diperbarui dalam {state['duration_seconds']}s",
            "state": state,
            "statuses": GLOBAL_CCTV_STATUS
        })
    else:
        t_async = threading.Thread(target=execute_full_refresh, daemon=True)
        t_async.start()
        return jsonify({
            "success": True,
            "message": "Pemindaian status CCTV dimulai di latar belakang",
            "state": REFRESH_STATE,
            "statuses": GLOBAL_CCTV_STATUS
        })


@app.route("/api/refresh-state")
def api_refresh_state():
    """Get the current scanning/refresh state."""
    return jsonify(REFRESH_STATE)


# ==========================================
# Streaming Proxy with Validated, Single-Flight .ts Cache
# ==========================================
SEGMENT_LOCKS_LOCK = threading.Lock()
SEGMENT_LOCKS = {}


def segment_lock(key: str) -> threading.Lock:
    """Single-flight lock per segment key so N viewers cause 1 upstream fetch."""
    with SEGMENT_LOCKS_LOCK:
        lock = SEGMENT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            SEGMENT_LOCKS[key] = lock
        if len(SEGMENT_LOCKS) > 512:
            for stale in [k for k, v in list(SEGMENT_LOCKS.items()) if k != key and not v.locked()][:256]:
                SEGMENT_LOCKS.pop(stale, None)
    return lock


def cache_get(key: str):
    with STREAM_CACHE_LOCK:
        entry = STREAM_CACHE.get(key)
        if entry is None:
            return None
        body, stored_at = entry
        if time.time() - stored_at >= CACHE_TTL_SECONDS:
            STREAM_CACHE.pop(key, None)
            return None
        STREAM_CACHE.move_to_end(key)
        return body


def cache_put(key: str, body: bytes) -> None:
    with STREAM_CACHE_LOCK:
        STREAM_CACHE[key] = (body, time.time())
        STREAM_CACHE.move_to_end(key)
        while len(STREAM_CACHE) > MAX_CACHE_ITEMS:
            STREAM_CACHE.popitem(last=False)


def segment_is_complete(body: bytes, declared_len) -> bool:
    """A cacheable MPEG-TS segment: non-empty, starts with sync byte 0x47, full length."""
    if not body or body[:1] != b"\x47":
        return False
    return declared_len is not None and len(body) == declared_len


def declared_length(res):
    raw = res.headers.get("Content-Length")
    return int(raw) if raw and raw.isdigit() else None


# ------------------------------------------------------------------
# Upstream allowlist: the proxy only talks to hosts that serve cameras in the catalog.
# Without this, anyone could make the server fetch internal addresses (SSRF).
# ------------------------------------------------------------------
MAX_UPSTREAM_REDIRECTS = 3


class BlockedUpstream(requests.RequestException):
    """Target (or redirect target) is not a catalog camera host."""


def url_origin(url: str):
    """(scheme, host, port) of a plain http(s) URL, or None if it must not be fetched.

    Parsed with urllib3, the same parser requests connects with, so the checked host is the real one.
    """
    if not url or "\\" in url or any(ord(ch) <= 32 for ch in url):
        return None
    try:
        parsed = parse_url(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or parsed.auth is not None or not parsed.host:
        return None
    return parsed.scheme, parsed.host.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


def allowed_origins() -> set:
    origins = {url_origin(m.get("stream", "")) for m in markers_global}
    origins.add(url_origin(UPSTREAM_VIDEO_BASE))
    origins.discard(None)
    return origins


def is_allowed_upstream(url: str) -> bool:
    origin = url_origin(url)
    return origin is not None and origin in allowed_origins()


def fetch_upstream(url: str):
    """GET a camera URL, checking the allowlist on the first request and on every redirect."""
    for _ in range(MAX_UPSTREAM_REDIRECTS + 1):
        if not is_allowed_upstream(url):
            raise BlockedUpstream("alamat di luar daftar kamera")
        res = UPSTREAM_SESSION.get(
            url, timeout=(UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_READ_TIMEOUT), allow_redirects=False
        )
        if res.status_code not in (301, 302, 303, 307, 308) or not res.headers.get("Location"):
            return res
        url = urllib.parse.urljoin(url, res.headers["Location"])
    raise BlockedUpstream("terlalu banyak pengalihan")


def blocked_response() -> Response:
    return Response("Alamat stream tidak diizinkan", status=403, headers=CORS_HEADERS)


def serve_playlist(target_url: str) -> Response:
    try:
        res = fetch_upstream(target_url)
    except BlockedUpstream:
        return blocked_response()
    except requests.RequestException as exc:
        return Response(f"Stream proxy connection error: {exc}", status=502, headers=CORS_HEADERS)
    if res.status_code != 200:
        return Response(f"Upstream stream error: {res.status_code}", status=res.status_code, headers=CORS_HEADERS)
    headers = dict(CORS_HEADERS)
    headers["Content-Type"] = "application/vnd.apple.mpegurl"
    headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return Response(res.content, status=200, headers=headers)


def segment_response(body: bytes, cache_state: str) -> Response:
    headers = dict(CORS_HEADERS)
    headers["Content-Type"] = "video/MP2T"
    headers["Cache-Control"] = "no-store"
    headers["X-Cache"] = cache_state
    return Response(body, status=200, headers=headers)


def serve_segment(cache_key: str, target_url: str) -> Response:
    cached = cache_get(cache_key)
    if cached is not None:
        return segment_response(cached, "HIT")

    with segment_lock(cache_key):
        cached = cache_get(cache_key)
        if cached is not None:
            return segment_response(cached, "HIT")

        body = b""
        last_error = ""
        for attempt in range(2):
            try:
                res = fetch_upstream(target_url)
            except BlockedUpstream:
                return blocked_response()
            except requests.RequestException as exc:
                last_error = str(exc)
                continue
            if res.status_code != 200:
                return Response(
                    f"Upstream stream error: {res.status_code}", status=res.status_code, headers=CORS_HEADERS
                )
            body = res.content
            if segment_is_complete(body, declared_length(res)):
                cache_put(cache_key, body)
                return segment_response(body, "MISS")
            if attempt == 0:
                time.sleep(0.4)

        if body[:1] == b"\x47":
            # Valid TS but truncated/unverifiable: serve it, never cache it.
            return segment_response(body, "PARTIAL")
        return Response(
            f"Upstream segment not ready: {last_error}", status=504, headers=CORS_HEADERS
        )


@app.route("/api/stream/<path:subpath>")
def proxy_cctv_stream(subpath):
    """Proxy live HLS m3u8 playlist and ts chunks with a short, validated segment cache."""
    target_url = urllib.parse.urljoin(UPSTREAM_VIDEO_BASE, subpath)
    # urljoin lets an absolute URL or "../" replace the base, so check the result stays inside it
    if not target_url.startswith(UPSTREAM_VIDEO_BASE) or ".." in urllib.parse.unquote(subpath).split("/"):
        return blocked_response()
    if subpath.endswith(".ts"):
        return serve_segment(subpath, target_url)
    return serve_playlist(target_url)


@app.route("/api/proxy_stream")
def proxy_generic_stream():
    """Proxy an m3u8 stream from a catalog camera host, with relative link rewriting."""
    raw_url = request.args.get("url", "").strip()
    if not raw_url:
        return Response("Parameter 'url' diperlukan", status=400, headers=CORS_HEADERS)

    if raw_url.startswith("https://pelindung.bandung.go.id/video/"):
        raw_url = raw_url.replace(
            "https://pelindung.bandung.go.id/video/", "https://pelindung.bandung.go.id:3443/video/"
        )

    if not is_allowed_upstream(raw_url):
        return blocked_response()

    if raw_url.endswith(".ts"):
        return serve_segment(raw_url, raw_url)

    try:
        res = fetch_upstream(raw_url)
    except BlockedUpstream:
        return blocked_response()
    except requests.RequestException as exc:
        return Response(f"Stream proxy connection error: {exc}", status=502, headers=CORS_HEADERS)
    if res.status_code != 200:
        return Response(f"Upstream stream error: {res.status_code}", status=res.status_code, headers=CORS_HEADERS)

    headers = dict(CORS_HEADERS)
    if raw_url.endswith(".m3u8"):
        headers["Content-Type"] = "application/vnd.apple.mpegurl"
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        base_url = raw_url.rsplit("/", 1)[0] + "/"
        rewritten = []
        for line in res.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                full_seg = urllib.parse.urljoin(base_url, stripped)
                rewritten.append(f"/api/proxy_stream?url={urllib.parse.quote(full_seg, safe='')}")
            else:
                rewritten.append(line)
        return Response("\n".join(rewritten).encode("utf-8"), status=200, headers=headers)

    headers["Content-Type"] = res.headers.get("Content-Type", "application/octet-stream")
    return Response(res.content, status=200, headers=headers)


@app.route("/api/rtsp_mjpeg")
def proxy_rtsp_mjpeg():
    """Stream an RTSP camera as multipart MJPEG using OpenCV so browsers can render it."""
    if cv2 is None:
        return Response("OpenCV tidak tersedia untuk streaming RTSP", status=503, headers=CORS_HEADERS)

    cctv_id = request.args.get("id", "").strip()
    rtsp_url = RTSP_URL_BY_ID.get(cctv_id)
    if not rtsp_url:
        return Response("Kamera RTSP tidak ditemukan", status=404, headers=CORS_HEADERS)

    def generate_frames():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        try:
            if not cap.isOpened():
                return
            consecutive_failures = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 15:
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                encoded_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not encoded_ok:
                    continue
                payload = jpg.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
                    + payload + b"\r\n"
                )
        finally:
            cap.release()

    headers = {
        **CORS_HEADERS,
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
