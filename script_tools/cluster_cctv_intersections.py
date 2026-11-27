#!/usr/bin/env python3
"""
Cluster and Group CCTV cameras around OpenStreetMap (OSM) road intersections (Perempatan / Persimpangan Jalan).

Features:
- Fetches and parses OpenStreetMap highway network data via Overpass API.
- Builds topological road graph to detect road intersections (Perempatan 4-way, Pertigaan 3-way, Simpang 5+).
- Merges multi-node junction complexes (e.g. dual-carriageway intersections) within proximity.
- Matches each CCTV to its nearest Perempatan and general intersection.
- Generates structured CSV files for analysis and reporting.

Usage:
    python cluster_cctv_intersections.py
    python cluster_cctv_intersections.py --radius 100 --input ../csv/cctv_bandung_data_validated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

DEFAULT_INPUT_CSV = Path("csv/cctv_bandung_data_validated.csv")
DEFAULT_OUTPUT_DIR = Path("csv")
DEFAULT_CACHE_FILE = Path("csv/osm_bandung_roads_cache.json")
DEFAULT_RADIUS = 100.0  # meters


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def parse_location(value: str) -> Optional[Tuple[float, float]]:
    """Parse 'lat,lon' string into float tuple."""
    parts = [p.strip() for p in (value or "").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except ValueError:
        pass
    return None


def load_cctv_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CCTV dataset from CSV."""
    if not csv_path.exists():
        # Fallback to parent/local search
        alt1 = Path("csv") / csv_path.name
        alt2 = Path("../csv") / csv_path.name
        if alt1.exists():
            csv_path = alt1
        elif alt2.exists():
            csv_path = alt2
        else:
            raise FileNotFoundError(f"CCTV CSV file not found: {csv_path}")

    cctvs: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            loc = parse_location(row.get("Location", ""))
            if loc is None:
                continue
            lat, lon = loc
            cctvs.append(
                {
                    "name": row.get("CCTV_Name", "").strip(),
                    "lat": lat,
                    "lon": lon,
                    "provider": row.get("Provider", "").strip(),
                    "m3u8_url": row.get("m3u8_URL", "").strip(),
                    "sample_downloaded": row.get("Sample_Downloaded", "").strip(),
                    "sample_file": row.get("Sample_File", "").strip(),
                    "sample_error": row.get("Sample_Error", "").strip(),
                    "raw_row": row,
                }
            )
    return cctvs


def fetch_osm_road_network(
    bbox: Tuple[float, float, float, float],
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch road ways and nodes from OpenStreetMap via Overpass API for a given bounding box.
    bbox: (min_lat, min_lon, max_lat, max_lon)
    """
    if (
        cache_path
        and cache_path.exists()
        and not force_refresh
        and cache_path.stat().st_size > 10000
    ):
        print(
            f"[OSM] Memuat data cache OpenStreetMap dari {cache_path} ({cache_path.stat().st_size / (1024*1024):.2f} MB)..."
        )
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    min_lat, min_lon, max_lat, max_lon = bbox
    overpass_query = f"""
[out:json][timeout:180];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service)$"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out body;
>;
out skel qt;
"""
    headers = {
        "User-Agent": "BandungCCTVIntersectionAnalysis/1.0 (https://github.com/CettaMa/cctv_weather)"
    }

    print(
        f"[OSM] Mengunduh jaringan jalan OpenStreetMap untuk BBox ({min_lat:.4f}, {min_lon:.4f}, {max_lat:.4f}, {max_lon:.4f})..."
    )
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"[OSM] Menghubungi endpoint: {endpoint}...")
            resp = requests.post(
                endpoint, data={"data": overpass_query}, headers=headers, timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"[OSM] Berhasil mengunduh {len(data.get('elements', []))} elemen dari OSM!"
                )
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f)
                    print(f"[OSM] Data disimpan ke cache: {cache_path}")
                return data
            else:
                print(f"[OSM] Endpoint mengembalikan status code: {resp.status_code}")
        except Exception as e:
            print(f"[OSM] Gagal terhubung ke {endpoint}: {e}")
            last_err = e

    raise RuntimeError(
        f"Gagal mengunduh data dari semua endpoint Overpass API: {last_err}"
    )


def extract_intersections_from_osm(
    osm_data: Dict[str, Any], cluster_distance: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Parse OSM elements, build road network graph, extract intersection nodes,
    and cluster close multi-node junctions into unified intersections.
    """
    nodes_dict: Dict[int, Tuple[float, float, Dict[str, str]]] = {}
    ways_list: List[Dict[str, Any]] = []

    for el in osm_data.get("elements", []):
        if el["type"] == "node":
            nodes_dict[el["id"]] = (el["lat"], el["lon"], el.get("tags", {}))
        elif el["type"] == "way":
            ways_list.append(el)

    print(
        f"[Analisis] Total OSM Nodes: {len(nodes_dict):,}, Total OSM Ways: {len(ways_list):,}"
    )

    node_neighbors: Dict[int, Set[int]] = defaultdict(set)
    node_road_names: Dict[int, Set[str]] = defaultdict(set)
    node_highway_types: Dict[int, Set[str]] = defaultdict(set)

    for way in ways_list:
        w_nodes = way.get("nodes", [])
        w_tags = way.get("tags", {})
        w_name = w_tags.get("name")
        w_ref = w_tags.get("ref")
        road_label = w_name if w_name else (w_ref if w_ref else "")
        hw_type = w_tags.get("highway", "road")

        for i, nid in enumerate(w_nodes):
            if nid not in nodes_dict:
                continue
            if road_label:
                node_road_names[nid].add(road_label)
            node_highway_types[nid].add(hw_type)

            if i > 0:
                prev_nid = w_nodes[i - 1]
                if prev_nid in nodes_dict:
                    node_neighbors[nid].add(prev_nid)
                    node_neighbors[prev_nid].add(nid)

    # Filter candidate intersection nodes (degree >= 3)
    cand_nodes: List[Dict[str, Any]] = []
    for nid, neighbors in node_neighbors.items():
        if len(neighbors) >= 3:
            lat, lon, tags = nodes_dict[nid]
            cand_nodes.append(
                {
                    "id": nid,
                    "lat": lat,
                    "lon": lon,
                    "degree": len(neighbors),
                    "neighbors": list(neighbors),
                    "roads": set(node_road_names[nid]),
                    "hw_types": set(node_highway_types[nid]),
                    "tags": tags,
                }
            )

    print(
        f"[Analisis] Ditemukan {len(cand_nodes):,} titik simpul persimpangan mentah (degree >= 3)."
    )

    # Spatial clustering (grouping nodes within cluster_distance, e.g. 25 meters)
    grid_size = 0.0003  # ~33m cell
    grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for item in cand_nodes:
        gx = int(item["lat"] / grid_size)
        gy = int(item["lon"] / grid_size)
        grid[(gx, gy)].append(item)

    visited: Set[int] = set()
    clusters: List[List[Dict[str, Any]]] = []

    for item in cand_nodes:
        if item["id"] in visited:
            continue
        cluster = [item]
        visited.add(item["id"])
        queue = [item]

        while queue:
            curr = queue.pop(0)
            cgx = int(curr["lat"] / grid_size)
            cgy = int(curr["lon"] / grid_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other in grid.get((cgx + dx, cgy + dy), []):
                        if other["id"] not in visited:
                            dist = haversine_distance(
                                curr["lat"], curr["lon"], other["lat"], other["lon"]
                            )
                            if dist <= cluster_distance:
                                visited.add(other["id"])
                                cluster.append(other)
                                queue.append(other)
        clusters.append(cluster)

    # Build consolidated intersection objects
    intersections: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        c_lat = sum(n["lat"] for n in cluster) / len(cluster)
        c_lon = sum(n["lon"] for n in cluster) / len(cluster)
        all_roads: Set[str] = set()
        all_hw: Set[str] = set()
        all_node_ids: Set[int] = set()
        has_traffic_signals = False

        for n in cluster:
            all_roads.update(n["roads"])
            all_hw.update(n["hw_types"])
            all_node_ids.add(n["id"])
            if (
                n["tags"].get("highway") == "traffic_signals"
                or n["tags"].get("crossing") == "traffic_signals"
            ):
                has_traffic_signals = True

        # Compute distinct external branches connecting to this intersection cluster
        external_neighbors: Set[int] = set()
        for n in cluster:
            for nbr in n["neighbors"]:
                if nbr not in all_node_ids:
                    external_neighbors.add(nbr)

        num_branches = len(external_neighbors)

        # Classification
        if num_branches == 4:
            itype = "Perempatan (4-way)"
            is_perempatan = True
        elif num_branches == 3:
            itype = "Pertigaan (3-way)"
            is_perempatan = False
        elif num_branches >= 5:
            itype = f"Simpang {num_branches} ({num_branches}-way)"
            is_perempatan = False
        elif num_branches == 2:
            itype = "Simpang Jalur Ganda (2-way)"
            is_perempatan = False
        else:
            itype = f"Simpang ({num_branches}-way)"
            is_perempatan = False

        sorted_roads = sorted(list(all_roads))
        if sorted_roads:
            road_name_str = " - ".join(sorted_roads)
            full_title = f"{itype.split(' ')[0]} {road_name_str}"
        else:
            road_name_str = "Tanpa Nama"
            full_title = f"{itype.split(' ')[0]} (Tanpa Nama)"

        intersections.append(
            {
                "intersection_id": f"INT-{idx + 1:05d}",
                "lat": c_lat,
                "lon": c_lon,
                "num_nodes": len(cluster),
                "osm_node_ids": sorted(list(all_node_ids)),
                "num_branches": num_branches,
                "type": itype,
                "is_perempatan": is_perempatan,
                "road_names": sorted_roads,
                "road_name_str": road_name_str,
                "full_name": full_title,
                "highway_types": sorted(list(all_hw)),
                "has_traffic_signals": has_traffic_signals,
            }
        )

    return intersections


def match_cctvs_to_intersections(
    cctv_list: List[Dict[str, Any]],
    intersections: List[Dict[str, Any]],
    radius_meters: float = 100.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match each CCTV to the nearest perempatan and nearest intersection in general.
    Also group intersections that have CCTVs within radius_meters.
    """
    cell_size = 0.005  # ~550m
    grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for it in intersections:
        gx = int(it["lat"] / cell_size)
        gy = int(it["lon"] / cell_size)
        grid[(gx, gy)].append(it)

    cctv_matches: List[Dict[str, Any]] = []
    intersection_cctv_groups: Dict[str, List[Tuple[Dict[str, Any], float]]] = (
        defaultdict(list)
    )

    perempatan_list = [it for it in intersections if it["is_perempatan"]]

    for cctv in cctv_list:
        cgx = int(cctv["lat"] / cell_size)
        cgy = int(cctv["lon"] / cell_size)

        candidates: List[Dict[str, Any]] = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                candidates.extend(grid.get((cgx + dx, cgy + dy), []))

        if not candidates:
            candidates = intersections

        nearest_perempatan = None
        min_p_dist = 9999999.0
        nearest_any = None
        min_any_dist = 9999999.0

        for it in candidates:
            dist = haversine_distance(cctv["lat"], cctv["lon"], it["lat"], it["lon"])
            if dist < min_any_dist:
                min_any_dist = dist
                nearest_any = it
            if it["is_perempatan"] and dist < min_p_dist:
                min_p_dist = dist
                nearest_perempatan = it

        if nearest_perempatan is None and perempatan_list:
            for it in perempatan_list:
                dist = haversine_distance(
                    cctv["lat"], cctv["lon"], it["lat"], it["lon"]
                )
                if dist < min_p_dist:
                    min_p_dist = dist
                    nearest_perempatan = it

        match_info = {
            "cctv": cctv,
            "nearest_perempatan": nearest_perempatan,
            "dist_perempatan_m": round(min_p_dist, 2),
            "is_near_perempatan": (min_p_dist <= radius_meters),
            "nearest_any_intersection": nearest_any,
            "dist_any_intersection_m": round(min_any_dist, 2),
            "is_near_any_intersection": (min_any_dist <= radius_meters),
        }
        cctv_matches.append(match_info)

        if nearest_perempatan and min_p_dist <= radius_meters:
            intersection_cctv_groups[nearest_perempatan["intersection_id"]].append(
                (cctv, min_p_dist)
            )

    intersection_dict = {it["intersection_id"]: it for it in intersections}
    cluster_rows: List[Dict[str, Any]] = []

    for int_id, cctv_pairs in intersection_cctv_groups.items():
        inter = intersection_dict[int_id]
        cctvs_in_cluster = [pair[0] for pair in cctv_pairs]
        dists = [pair[1] for pair in cctv_pairs]

        cctv_names = [c["name"] for c in cctvs_in_cluster]
        cctv_providers = sorted(list(set(c["provider"] for c in cctvs_in_cluster)))
        cctv_streams = [c["m3u8_url"] for c in cctvs_in_cluster if c["m3u8_url"]]

        cluster_rows.append(
            {
                "intersection_id": inter["intersection_id"],
                "intersection_name": inter["full_name"],
                "intersection_type": inter["type"],
                "road_names": inter["road_name_str"],
                "latitude": inter["lat"],
                "longitude": inter["lon"],
                "has_traffic_signals": "YES" if inter["has_traffic_signals"] else "NO",
                "cctv_count": len(cctvs_in_cluster),
                "cctv_names": " | ".join(cctv_names),
                "cctv_providers": ", ".join(cctv_providers),
                "min_distance_m": round(min(dists), 2),
                "max_distance_m": round(max(dists), 2),
                "avg_distance_m": round(sum(dists) / len(dists), 2),
                "cctv_stream_urls": " | ".join(cctv_streams),
                "osm_node_ids": ",".join(str(nid) for nid in inter["osm_node_ids"]),
                "cctvs_list": cctvs_in_cluster,
            }
        )

    cluster_rows.sort(key=lambda x: (-x["cctv_count"], x["min_distance_m"]))
    return cctv_matches, cluster_rows


def export_csv_files(
    cctv_matches: List[Dict[str, Any]],
    cluster_rows: List[Dict[str, Any]],
    all_intersections: List[Dict[str, Any]],
    output_dir: Path,
    radius_meters: float,
) -> Dict[str, Path]:
    """Write all structured CSV files to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    # 1. Perempatan Clusters CSV
    clusters_csv = output_dir / "cctv_perempatan_clusters.csv"
    with clusters_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "Intersection_ID",
            "Intersection_Name",
            "Intersection_Type",
            "Road_Names",
            "Latitude",
            "Longitude",
            "Has_Traffic_Signals",
            "CCTV_Count",
            "CCTV_Names",
            "CCTV_Providers",
            "Min_Distance_Meters",
            "Max_Distance_Meters",
            "Avg_Distance_Meters",
            "CCTV_Stream_URLs",
            "OSM_Node_IDs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cluster_rows:
            writer.writerow(
                {
                    "Intersection_ID": row["intersection_id"],
                    "Intersection_Name": row["intersection_name"],
                    "Intersection_Type": row["intersection_type"],
                    "Road_Names": row["road_names"],
                    "Latitude": f"{row['latitude']:.7f}",
                    "Longitude": f"{row['longitude']:.7f}",
                    "Has_Traffic_Signals": row["has_traffic_signals"],
                    "CCTV_Count": row["cctv_count"],
                    "CCTV_Names": row["cctv_names"],
                    "CCTV_Providers": row["cctv_providers"],
                    "Min_Distance_Meters": row["min_distance_m"],
                    "Max_Distance_Meters": row["max_distance_m"],
                    "Avg_Distance_Meters": row["avg_distance_m"],
                    "CCTV_Stream_URLs": row["cctv_stream_urls"],
                    "OSM_Node_IDs": row["osm_node_ids"],
                }
            )
    generated_files["clusters"] = clusters_csv

    # 2. Detailed CCTV with Nearest Perempatan
    cctv_perempatan_csv = output_dir / "cctv_perempatan_detail.csv"
    with cctv_perempatan_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "CCTV_Name",
            "CCTV_Latitude",
            "CCTV_Longitude",
            "Provider",
            "Nearest_Perempatan_ID",
            "Nearest_Perempatan_Name",
            "Intersecting_Roads",
            "Perempatan_Latitude",
            "Perempatan_Longitude",
            "Distance_to_Perempatan_Meters",
            f"Within_Radius_{int(radius_meters)}m",
            "Has_Traffic_Signals",
            "m3u8_URL",
            "Sample_Downloaded",
            "Sample_File",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in cctv_matches:
            c = m["cctv"]
            p = m["nearest_perempatan"]
            writer.writerow(
                {
                    "CCTV_Name": c["name"],
                    "CCTV_Latitude": f"{c['lat']:.7f}",
                    "CCTV_Longitude": f"{c['lon']:.7f}",
                    "Provider": c["provider"],
                    "Nearest_Perempatan_ID": p["intersection_id"] if p else "",
                    "Nearest_Perempatan_Name": p["full_name"] if p else "",
                    "Intersecting_Roads": p["road_name_str"] if p else "",
                    "Perempatan_Latitude": f"{p['lat']:.7f}" if p else "",
                    "Perempatan_Longitude": f"{p['lon']:.7f}" if p else "",
                    "Distance_to_Perempatan_Meters": m["dist_perempatan_m"],
                    f"Within_Radius_{int(radius_meters)}m": (
                        "YES" if m["is_near_perempatan"] else "NO"
                    ),
                    "Has_Traffic_Signals": (
                        "YES" if (p and p["has_traffic_signals"]) else "NO"
                    ),
                    "m3u8_URL": c["m3u8_url"],
                    "Sample_Downloaded": c["sample_downloaded"],
                    "Sample_File": c["sample_file"],
                }
            )
    generated_files["perempatan_detail"] = cctv_perempatan_csv

    # 3. Only CCTVs within Radius of Perempatan
    cctv_perempatan_filtered_csv = (
        output_dir / f"cctv_di_sekitar_perempatan_{int(radius_meters)}m.csv"
    )
    with cctv_perempatan_filtered_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fieldnames = [
            "CCTV_Name",
            "CCTV_Latitude",
            "CCTV_Longitude",
            "Provider",
            "Nearest_Perempatan_ID",
            "Nearest_Perempatan_Name",
            "Intersecting_Roads",
            "Perempatan_Latitude",
            "Perempatan_Longitude",
            "Distance_to_Perempatan_Meters",
            "Has_Traffic_Signals",
            "m3u8_URL",
            "Sample_Downloaded",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in cctv_matches:
            if not m["is_near_perempatan"]:
                continue
            c = m["cctv"]
            p = m["nearest_perempatan"]
            writer.writerow(
                {
                    "CCTV_Name": c["name"],
                    "CCTV_Latitude": f"{c['lat']:.7f}",
                    "CCTV_Longitude": f"{c['lon']:.7f}",
                    "Provider": c["provider"],
                    "Nearest_Perempatan_ID": p["intersection_id"] if p else "",
                    "Nearest_Perempatan_Name": p["full_name"] if p else "",
                    "Intersecting_Roads": p["road_name_str"] if p else "",
                    "Perempatan_Latitude": f"{p['lat']:.7f}" if p else "",
                    "Perempatan_Longitude": f"{p['lon']:.7f}" if p else "",
                    "Distance_to_Perempatan_Meters": m["dist_perempatan_m"],
                    "Has_Traffic_Signals": (
                        "YES" if (p and p["has_traffic_signals"]) else "NO"
                    ),
                    "m3u8_URL": c["m3u8_url"],
                    "Sample_Downloaded": c["sample_downloaded"],
                }
            )
    generated_files["perempatan_filtered"] = cctv_perempatan_filtered_csv

    # 4. Master List of All Perempatan in Bandung from OpenStreetMap
    perempatan_master_csv = output_dir / "osm_bandung_semua_perempatan.csv"
    with perempatan_master_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "Intersection_ID",
            "Intersection_Type",
            "Road_Names",
            "Latitude",
            "Longitude",
            "Num_Branches",
            "Has_Traffic_Signals",
            "Highway_Types",
            "OSM_Node_IDs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in all_intersections:
            if not it["is_perempatan"]:
                continue
            writer.writerow(
                {
                    "Intersection_ID": it["intersection_id"],
                    "Intersection_Type": it["type"],
                    "Road_Names": it["road_name_str"],
                    "Latitude": f"{it['lat']:.7f}",
                    "Longitude": f"{it['lon']:.7f}",
                    "Num_Branches": it["num_branches"],
                    "Has_Traffic_Signals": "YES" if it["has_traffic_signals"] else "NO",
                    "Highway_Types": ", ".join(it["highway_types"]),
                    "OSM_Node_IDs": ",".join(str(nid) for nid in it["osm_node_ids"]),
                }
            )
    generated_files["osm_perempatan_master"] = perempatan_master_csv

    # 5. All Intersections Detail (Perempatan, Pertigaan, Simpang 5+)
    all_persimpangan_csv = output_dir / "cctv_semua_persimpangan_detail.csv"
    with all_persimpangan_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "CCTV_Name",
            "CCTV_Latitude",
            "CCTV_Longitude",
            "Provider",
            "Nearest_Intersection_ID",
            "Nearest_Intersection_Name",
            "Intersection_Type",
            "Intersecting_Roads",
            "Intersection_Latitude",
            "Intersection_Longitude",
            "Distance_to_Intersection_Meters",
            f"Within_Radius_{int(radius_meters)}m",
            "Has_Traffic_Signals",
            "m3u8_URL",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in cctv_matches:
            c = m["cctv"]
            it = m["nearest_any_intersection"]
            writer.writerow(
                {
                    "CCTV_Name": c["name"],
                    "CCTV_Latitude": f"{c['lat']:.7f}",
                    "CCTV_Longitude": f"{c['lon']:.7f}",
                    "Provider": c["provider"],
                    "Nearest_Intersection_ID": it["intersection_id"] if it else "",
                    "Nearest_Intersection_Name": it["full_name"] if it else "",
                    "Intersection_Type": it["type"] if it else "",
                    "Intersecting_Roads": it["road_name_str"] if it else "",
                    "Intersection_Latitude": f"{it['lat']:.7f}" if it else "",
                    "Intersection_Longitude": f"{it['lon']:.7f}" if it else "",
                    "Distance_to_Intersection_Meters": m["dist_any_intersection_m"],
                    f"Within_Radius_{int(radius_meters)}m": (
                        "YES" if m["is_near_any_intersection"] else "NO"
                    ),
                    "Has_Traffic_Signals": (
                        "YES" if (it and it["has_traffic_signals"]) else "NO"
                    ),
                    "m3u8_URL": c["m3u8_url"],
                }
            )
    generated_files["semua_persimpangan_detail"] = all_persimpangan_csv

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster and Group CCTV cameras around OpenStreetMap road intersections."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CCTV CSV file (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help=f"Max distance (meters) to consider a CCTV as near an intersection (default: {DEFAULT_RADIUS}m)",
    )
    parser.add_argument(
        "--cache-osm",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help=f"Path to cache OSM data (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--refresh-osm",
        action="store_true",
        help="Force re-download OSM road network data from Overpass API",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PENGELOMPOKAN CCTV DI SEKITAR PEREMPATAN JALAN (OPENSTREETMAP)  ")
    print("=" * 70)

    # 1. Load CCTV Data
    print(f"\n[1/4] Membaca dataset CCTV dari: {args.input}")
    cctv_list = load_cctv_data(args.input)
    print(f"      Total CCTV valid dengan koordinat: {len(cctv_list)}")

    if not cctv_list:
        print("Error: Tidak ada data CCTV valid yang ditemukan.")
        sys.exit(1)

    # 2. Compute Bounding Box covering all CCTVs
    lats = [c["lat"] for c in cctv_list]
    lons = [c["lon"] for c in cctv_list]
    buffer_deg = 0.015
    bbox = (
        min(lats) - buffer_deg,
        min(lons) - buffer_deg,
        max(lats) + buffer_deg,
        max(lons) + buffer_deg,
    )
    print(
        f"      Area Bounding Box: Lat [{bbox[0]:.4f}, {bbox[2]:.4f}], Lon [{bbox[1]:.4f}, {bbox[3]:.4f}]"
    )

    # 3. Fetch & Parse OpenStreetMap Road Network
    print(f"\n[2/4] Mengambil & menganalisis data perempatan dari OpenStreetMap...")
    osm_data = fetch_osm_road_network(
        bbox, cache_path=args.cache_osm, force_refresh=args.refresh_osm
    )
    all_intersections = extract_intersections_from_osm(
        osm_data, cluster_distance=25.0
    )

    perempatan_list = [it for it in all_intersections if it["is_perempatan"]]
    pertigaan_list = [it for it in all_intersections if it["num_branches"] == 3]
    simpang5_list = [it for it in all_intersections if it["num_branches"] >= 5]

    print(f"\n[3/4] Statistik Persimpangan OpenStreetMap di Area:")
    print(f"      - Total Klaster Persimpangan: {len(all_intersections):,}")
    print(f"      - Perempatan (4-way crossroad): {len(perempatan_list):,}")
    print(f"      - Pertigaan (3-way T-junction): {len(pertigaan_list):,}")
    print(f"      - Simpang 5+ (5+ way junction): {len(simpang5_list):,}")

    # 4. Match CCTVs & Group by Intersection
    print(f"\n[4/4] Mencocokkan {len(cctv_list)} CCTV dengan radius <= {args.radius}m...")
    cctv_matches, cluster_rows = match_cctvs_to_intersections(
        cctv_list, all_intersections, radius_meters=args.radius
    )

    cctvs_near_perempatan = [m for m in cctv_matches if m["is_near_perempatan"]]
    cctvs_near_any = [m for m in cctv_matches if m["is_near_any_intersection"]]

    print(f"\n" + "=" * 70)
    print(f"  HASIL PENGELOMPOKAN CCTV:")
    print(f"  * CCTV di sekitar Perempatan (<= {args.radius}m): {len(cctvs_near_perempatan)} dari {len(cctv_list)} CCTV ({len(cctvs_near_perempatan)/len(cctv_list)*100:.1f}%)")
    print(f"  * CCTV di sekitar Persimpangan Apapun (<= {args.radius}m): {len(cctvs_near_any)} dari {len(cctv_list)} CCTV ({len(cctvs_near_any)/len(cctv_list)*100:.1f}%)")
    print(f"  * Jumlah Perempatan Aktif yang Memiliki CCTV: {len(cluster_rows)} lokasi")
    print("=" * 70)

    # 5. Export CSV Files
    generated_files = export_csv_files(
        cctv_matches, cluster_rows, all_intersections, args.output_dir, args.radius
    )

    print("\nFile CSV berhasil dibuat:")
    for key, path in generated_files.items():
        size_kb = path.stat().st_size / 1024
        print(f"  [+] {path.name:45s} ({size_kb:6.1f} KB) -> {path}")

    # Display Top 10 Perempatan with most CCTVs
    print("\n--- TOP 10 PEREMPATAN DENGAN CCTV TERBANYAK ---")
    for rank, row in enumerate(cluster_rows[:10], 1):
        print(
            f"{rank:2d}. {row['intersection_name']} ({row['cctv_count']} CCTV, Jarak Rata-rata: {row['avg_distance_m']}m)"
        )
        print(f"    CCTV: {row['cctv_names']}")
        print(f"    Lampu Lalu Lintas: {row['has_traffic_signals']} | Koordinat: {row['latitude']:.6f}, {row['longitude']:.6f}")


if __name__ == "__main__":
    main()
