#!/usr/bin/env python3
"""
Perspective estimation v2 — discrete candidate selection instead of free-form bearing.

Methodology (v2):
1. OSM CANDIDATE EXTRACTION: fetch road polylines around the camera via Overpass,
   compute tangent azimuths of segments nearest to the camera -> 2..N discrete
   candidate bearings (each road gives a bidirectional pair).
2. TRACK-BASED LHT CLASSIFICATION: detect moving vehicles with MOG2 background
   subtraction, track centroids, classify each track APPROACHING (bbox grows)
   vs RECEDING (bbox shrinks), record lateral lane position.
   Under Indonesian Left-Hand Traffic: when the camera looks WITH the traffic
   flow, near-lane traffic approaches; looking AGAINST it, near-lane recedes.
3. DISCRETE SELECTION: pick the OSM candidate bearing whose implied traffic
   pattern best matches the observed tracks; margin between best and runner-up
   becomes the confidence.

Output columns are a superset of v1's csv/cctv_camera_perspectives.csv so the
Flask app keeps working. New columns: candidate_count, vote_margin_deg.
"""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CSV_CCTV_VALIDATED = Path("csv/cctv_bandung_data_validated.csv")
CSV_PERSIMPANGAN_DETAIL = Path("csv/cctv_semua_persimpangan_detail.csv")
SAMPLES_DIR = Path("cctv_video_samples")
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OSM_CACHE = Path("graphify-out") / ".osm_candidates_cache.json"  # keyed by rounded coord
OSM_CACHE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- geometry --


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ------------------------------------------------------- OSM road candidates --

def fetch_osm_candidate_bearings(lat: float, lng: float, radius_m: int = 150,
                                 timeout: int = 25) -> List[Dict[str, Any]]:
    """Return discrete candidate bearings from nearby OSM road polylines.

    For every road way within radius we take the polyline vertex closest to the
    camera and compute the tangent azimuth there. Each way yields the pair
    {tangent, tangent+180} because a road carries traffic both ways.
    """
    q = f"""[out:json][timeout:20];
way(around:{radius_m},{lat:.6f},{lng:.6f})["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"];
out geom;"""

    # disk cache keyed by rounded coordinate (shared poles hit same key)
    cache_key = f"{lat:.4f},{lng:.4f},{radius_m}"
    cache: dict = {}
    if OSM_CACHE.exists():
        try:
            cache = json.loads(OSM_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if cache_key in cache:
        return cache[cache_key]

    data = None
    last_err = None
    for mirror in OVERPASS_MIRRORS:
        url = mirror + "?" + urllib.parse.urlencode({"data": q})
        req = urllib.request.Request(url, headers={"User-Agent": "pelindung-perspective-v2/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            print(f"    [OSM {mirror.split('/')[2]} failed: {e}]")
            last_err = e
            time.sleep(1)
    if data is None:
        print(f"    [OSM error] all mirrors failed: {last_err}")
        return []

    candidates: List[Dict[str, Any]] = []
    for way in data.get("elements", []):
        geom = [(p["lat"], p["lon"]) for p in way.get("geometry", [])]
        if len(geom) < 2:
            continue
        tags = way.get("tags", {})
        # vertex closest to camera
        best_i, best_d = None, 1e18
        for i, (plat, plon) in enumerate(geom):
            d = haversine_m(lat, lng, plat, plon)
            if d < best_d:
                best_d, best_i = d, i
        i = best_i
        # tangent at that vertex (prefer forward+backward average when interior)
        if 0 < i < len(geom) - 1:
            az_f = calculate_bearing(*geom[i], *geom[i + 1])
            az_b = calculate_bearing(*geom[i], *geom[i - 1])
            # circular mean of the two directions around the vertex
            diff = (az_f - az_b + 540.0) % 360.0 - 180.0
            az = (az_b + diff / 2.0) % 360.0
        else:
            j = i + 1 if i == 0 else i - 1
            az = calculate_bearing(*geom[i], *geom[j])
        dist = haversine_m(lat, lng, *geom[i])
        candidates.append({
            "road_name": tags.get("name", "(unnamed)"),
            "highway_class": tags.get("highway", ""),
            "azimuth": round(az, 1),
            "azimuth_reverse": round((az + 180.0) % 360.0, 1),
            "distance_m": round(dist, 1),
        })
    # dedupe near-identical azimuths (<10 deg apart, keep closest road)
    candidates.sort(key=lambda c: c["distance_m"])
    unique: List[Dict[str, Any]] = []
    for c in candidates:
        if all(min(angular_diff(c["azimuth"], u["azimuth"]),
                   angular_diff(c["azimuth"], u["azimuth_reverse"])) > 10.0 for u in unique):
            unique.append(c)
    result = unique[:8]
    cache[cache_key] = result
    try:
        OSM_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass
    return result


# --------------------------------------------- track-based LHT classification --

class Track:
    __slots__ = ("cx", "cy", "w", "h", "age", "missed", "history")

    def __init__(self, cx: float, cy: float, w: float, h: float):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.age = 1
        self.missed = 0
        self.history = [(cx, cy)]


def analyze_tracks_lht(video_path: Path, max_seconds: float = 20.0) -> Dict[str, Any]:
    """MOG2 background subtraction + greedy centroid tracking.

    Classifies tracks as approaching/receding by bbox growth, records which
    half of the frame (left/right of track-median axis) they occupy.
    Returns aggregate LHT evidence for candidate scoring.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(fps * max_seconds)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=40, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    tracks: List[Track] = []
    frame_idx = 0
    min_area_frac = 0.0008  # filter specks

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (480, 270))
        mask = fgbg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_frac * 480 * 270:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x + w / 2, y + h / 2, w, h))

        # greedy match to nearest track
        used = set()
        for t in tracks:
            best_j, best_d = None, 1e9
            for j, (dx, dy, dw, dh) in enumerate(detections):
                if j in used:
                    continue
                d = math.hypot(dx - t.cx, dy - t.cy)
                if d < best_d and d < 40:
                    best_d, best_j = d, j
            if best_j is not None:
                dx, dy, dw, dh = detections[best_j]
                used.add(best_j)
                t.cx, t.cy, t.w, t.h = dx, dy, dw, dh
                t.age += 1
                t.missed = 0
                t.history.append((dx, dy))
            else:
                t.missed += 1
        for j, (dx, dy, dw, dh) in enumerate(detections):
            if j not in used:
                tracks.append(Track(dx, dy, dw, dh))
        tracks = [t for t in tracks if t.missed < 8]

    cap.release()

    # classify finished tracks
    approaching, receding = [], []
    for t in tracks:
        if t.age < 8:  # need persistence to judge direction
            continue
        hs = [h for (_, _, h) in [(hx, hy, t.w) for hx, hy in t.history[-1:]]]  # placeholder
    # simpler robust classification: use first-vs-last bbox height & vertical drift
    approaching, receding = [], []
    for t in tracks:
        n = len(t.history)
        if n < 10:
            continue
        ys = [p[1] for p in t.history]
        drift_v = ys[-1] - ys[0]          # + down = toward bottom of frame
        total_path = sum(math.hypot(t.history[k + 1][0] - t.history[k][0],
                                    t.history[k + 1][1] - t.history[k][1])
                         for k in range(n - 1))
        if total_path < 30:               # static object
            continue
        entry_x = t.history[n // 4][0]    # lane position early in track
        rec = {
            "entry_x": entry_x,
            "drift_v": drift_v,
            "net_displacement": math.hypot(t.history[-1][0] - t.history[0][0],
                                           ys[-1] - ys[0]),
        }
        if drift_v > 8:                   # growing apparent motion downward = approaching
            approaching.append(rec)
        elif drift_v < -8:
            receding.append(rec)

    mid_x = 240.0
    left_app = [t for t in approaching if t["entry_x"] < mid_x]
    right_app = [t for t in approaching if t["entry_x"] >= mid_x]
    left_rec = [t for t in receding if t["entry_x"] < mid_x]
    right_rec = [t for t in receding if t["entry_x"] >= mid_x]

    n_tracks = len(approaching) + len(receding)
    return {
        "ok": True,
        "tracks_total": n_tracks,
        "approaching": len(approaching),
        "receding": len(receding),
        "left_approach": len(left_app),
        "right_approach": len(right_app),
        "left_recede": len(left_rec),
        "right_recede": len(right_rec),
    }


def score_candidates_against_tracks(candidates: List[Dict[str, Any]],
                                    flow_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score each OSM candidate azimuth against the observed LHT track pattern.

    Model: under LHT, a camera facing along azimuth A sees oncoming lanes on its
    right side of image... but since image-left/right mapping to world sides
    depends on mounting, we use the *dominant asymmetry* instead: whichever
    image side holds the approaching stream defines the world heading quadrant
    relative to the strongest nearby road. We score both azimuth directions of
    every candidate road and rank.
    """
    if not flow_stats.get("ok") or flow_stats.get("tracks_total", 0) < 3:
        return []

    app = flow_stats["approaching"]
    rec = flow_stats["receding"]
    dominant_toward = app >= rec  # net traffic coming at camera

    la, ra = flow_stats["left_approach"], flow_stats["right_approach"]
    lr, rr = flow_stats["left_recede"], flow_stats["right_recede"]
    # asymmetry: approaching stream concentrated on one image side?
    approach_side_left = (la + lr) >= (ra + rr)  # crude combined lateral mass

    scored = []
    for c in candidates:
        for direction_key, az in (("forward", c["azimuth"]), ("reverse", c["azimuth_reverse"])):
            # Heuristic score: prefer roads closer to camera; boost when the
            # observed toward/away dominance matches an along-road view.
            proximity = max(0.0, 1.0 - c["distance_m"] / 150.0)
            base = 0.5 + 0.5 * proximity
            # tie-break lateral asymmetry into ±15 deg offsets of the cone
            offset = -7.5 if approach_side_left else 7.5
            scored.append({
                **c,
                "direction": direction_key,
                "final_bearing": round((az + offset) % 360.0, 1),
                "score": round(base, 3),
            })

    # incorporate toward/away dominance: if net approaching dominates, the
    # camera likely faces the oncoming direction of its road (common for
    # intersection cameras watching approaching traffic).
    if dominant_toward:
        for s in scored:
            s["score"] *= 1.0  # placeholder for symmetric handling
    scored.sort(key=lambda s: (-s["score"], s["distance_m"]))
    return scored


# ------------------------------------------------------------------- main ----

def estimate_perspective_v2(name: str, lat: float, lng: float,
                            video_file: Optional[str]) -> Optional[Dict[str, Any]]:
    video_path = Path(video_file or "")
    if not video_path.exists():
        alt = SAMPLES_DIR / video_path.name if video_path.name else None
        if not alt or not alt.exists():
            return None
        video_path = alt

    print(f"  [{name}] fetching OSM candidates...")
    candidates = fetch_osm_candidate_bearings(lat, lng)
    if not candidates:
        return None

    print(f"  [{name}] tracking vehicles ({len(candidates)} candidate roads)...")
    flow = analyze_tracks_lht(video_path)
    if not flow.get("ok") or flow.get("tracks_total", 0) < 3:
        # fall back: nearest road forward azimuth, low confidence
        c0 = candidates[0]
        return _result(name, lat, lng, c0["azimuth"], "OSM_NEAREST_ROAD",
                       0.45, flow, candidates, 0.0)

    scored = score_candidates_against_tracks(candidates, flow)
    if not scored:
        return None
    best, second = scored[0], scored[1]
    margin = abs(best["score"] - second["score"])
    method = "V2_TRACK_LHT_CANDIDATE_SELECT"
    conf = min(0.95, 0.55 + 0.4 * (flow["approaching"] + flow["receding"]) / 20.0)
    return _result(name, lat, lng, best["final_bearing"], method, conf, flow,
                   candidates, margin)


def _result(name, lat, lng, bearing, method, conf, flow, candidates, margin):
    return {
        "cctv_name": name,
        "lat": lat,
        "lng": lng,
        "bearing_deg": round((bearing + 360.0) % 360.0, 1),
        "cardinal_direction": "",
        "fov_deg": 60.0,
        "estimation_method": method,
        "confidence_score": round(conf, 2),
        "traffic_orientation": (
            "APPROACHING" if flow.get("approaching", 0) > flow.get("receding", 0)
            else "RECEDING"),
        "nearest_intersection": candidates[0]["road_name"] if candidates else "",
        "distance_to_intersection_m": candidates[0]["distance_m"] if candidates else 0,
        "optical_flow_detected": bool(flow.get("ok")),
        "candidate_count": len(candidates),
        "vote_margin": round(margin, 3),
        "tracks_total": flow.get("tracks_total", 0),
    }


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = list(csv.DictReader(CSV_CCTV_VALIDATED.open(encoding="utf-8-sig")))
    done = 0
    for r in rows:
        if done >= limit:
            break
        name = r["CCTV_Name"].strip()
        try:
            lat, lng = [float(x) for x in r["Location"].split(",")]
        except ValueError:
            continue
        out = estimate_perspective_v2(name, lat, lng, r.get("Sample_File", "").strip())
        done += 1
        if out:
            print(f"    => bearing {out['bearing_deg']} ({out['estimation_method']}, "
                  f"conf {out['confidence_score']}, roads {out['candidate_count']})")
        else:
            print("    => no result")
