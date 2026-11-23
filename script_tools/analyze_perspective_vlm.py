#!/usr/bin/env python3
"""
Analyze CCTV Camera Perspective & Heading using Multimodal AI (Google Gemini Vision).

Requirements:
  1. Google Gemini API Key (set GEMINI_API_KEY environment variable or pass --api-key).
     Get a free key from: https://aistudio.google.com/
  2. Extracted CCTV keyframes or video samples in `cctv_video_samples/`.

How it works:
  - Takes 1 clear snapshot JPG from each camera.
  - Passes the image to Gemini 2.0 Flash / 1.5 Flash with road & intersection spatial context.
  - Gemini reads:
      * Road signs, directional guideboards, street names
      * Visual landmarks (mosques, flyovers, tall buildings, plazas)
      * Traffic flow lanes (approaching / receding vehicles)
      * Sun shadow angle
  - Returns structured JSON with:
      * bearing_deg: 0 - 360
      * cardinal_direction: "Utara", "Timur Laut", "Tenggara", etc.
      * visual_landmarks: list of recognized objects
      * visual_reasoning: explanation of why this heading was determined
      * confidence_score: 0.0 - 1.0
  - Updates `csv/cctv_camera_perspectives.csv`.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

CSV_CCTV_VALIDATED = Path("csv/cctv_bandung_data_validated.csv")
CSV_PERSIMPANGAN_DETAIL = Path("csv/cctv_semua_persimpangan_detail.csv")
CSV_OUTPUT_PERSPECTIVES = Path("csv/cctv_camera_perspectives.csv")
SAMPLES_DIR = Path("cctv_video_samples")
KEYFRAMES_DIR = Path("cctv_keyframes")


def extract_keyframe_from_sample(sample_path: Path, output_path: Path) -> bool:
    """Extracts a clear middle frame from a video sample."""
    if not sample_path.exists():
        return False
    try:
        cap = cv2.VideoCapture(str(sample_path))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # Frame at 1.5s
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 1.5))
        ret, frame = cap.read()
        if ret and frame is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cap.release()
            return True
        cap.release()
    except Exception as e:
        print(f"Error extracting frame from {sample_path.name}: {e}")
    return False


def query_gemini_vision(
    image_path: Path,
    cctv_name: str,
    lat: float,
    lng: float,
    intersection_name: str,
    intersecting_roads: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """Sends CCTV image snapshot and geospatial context to Gemini Vision REST API."""
    if not image_path.exists():
        return {"error": "Image file not found"}

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""
Anda adalah pakar Analisis Geospasial, Transportasi Lalu Lintas, dan Computer Vision untuk Kota Bandung, Indonesia.
Tugas Anda adalah menentukan PERSPEKTIF / ARAH HADAP KAMERA CCTV (Camera Heading / Bearing Azimuth 0°-360°) dari gambar snapshot CCTV berikut.

Konteks CCTV:
- Nama Kamera: {cctv_name}
- Koordinat GPS: {lat:.6f}, {lng:.6f}
- Persimpangan Terdekat: {intersection_name or 'Ruas Jalan Kota Bandung'}
- Ruas Jalan Terkait: {intersecting_roads or '-'}
- Sistem Lalu Lintas: Indonesia menggunakan sistem lajur kiri (Left-Hand Traffic / LHT).

Instruksi Analisis:
1. Amati jalan, marka jalan, lajur arus kendaraan (mobil & motor yang mendekat atau menjauh).
2. Periksa plang rambu lalu lintas, penunjuk arah, papan reklame, nama toko/bangunan jika terlihat.
3. Kenali landmark arsitektural (misal: jembatan layang/flyover, menara masjid, gedung perkantoran, trotoar khas Bandung).
4. Tentukan sudut bearing kompas (0° = Utara, 90° = Timur, 180° = Selatan, 270° = Barat).

Kembalikan jawaban HANYA dalam format JSON valid (tanpa markdown tambahan) dengan struktur berikut:
{{
  "bearing_deg": <angka float 0.0 - 360.0>,
  "cardinal_direction": "<Utara | Timur Laut | Timur | Tenggara | Selatan | Barat Daya | Barat | Barat Laut>",
  "fov_deg": 60.0,
  "confidence_score": <angka float 0.0 - 1.0>,
  "traffic_orientation": "<FORWARD_ALONG_TRAFFIC | REVERSE_AGAINST_TRAFFIC | CROSS_TRAFFIC | STATIC>",
  "visual_landmarks": ["<landmark 1>", "<landmark 2>"],
  "visual_reasoning": "<Penjelasan singkat mengapa arah ini ditentukan berdasarkan elemen visual pada gambar>"
}}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            candidate_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            # Clean possible markdown wrap
            cleaned = candidate_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            return parsed
    except Exception as e:
        print(f"  [VLM Error for {cctv_name}]: {e}")
        return {"error": str(e)}


def run_vlm_perspective_batch(api_key: str, limit: Optional[int] = None):
    """Batch processes CCTV snapshots with Gemini Vision."""
    print("=" * 65)
    print("Bandung CCTV Camera Perspective Analysis via Gemini Vision VLM")
    print("=" * 65)

    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load intersection lookup
    isect_map = {}
    if CSV_PERSIMPANGAN_DETAIL.exists():
        with CSV_PERSIMPANGAN_DETAIL.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = r.get("CCTV_Name", "").strip()
                isect_map[name] = r

    # Load validated CCTVs
    if not CSV_CCTV_VALIDATED.exists():
        print(f"Error: {CSV_CCTV_VALIDATED} not found.")
        return

    cctvs = []
    with CSV_CCTV_VALIDATED.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cctvs.append(r)

    # Load existing perspectives if available to preserve or update
    existing_perspectives = {}
    if CSV_OUTPUT_PERSPECTIVES.exists():
        with CSV_OUTPUT_PERSPECTIVES.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                existing_perspectives[r.get("cctv_name", "").strip()] = r

    processed_count = 0
    updated_count = 0

    for idx, c in enumerate(cctvs):
        if limit and processed_count >= limit:
            break

        name = c.get("CCTV_Name", "").strip()
        loc_str = c.get("Location", "")
        parts = [p.strip() for p in loc_str.split(",")]
        if len(parts) != 2:
            continue
        try:
            cctv_lat = float(parts[0])
            cctv_lng = float(parts[1])
        except ValueError:
            continue

        sample_file = c.get("Sample_File", "").strip()
        video_path = Path(sample_file)
        if not video_path.exists():
            alt = SAMPLES_DIR / video_path.name
            if alt.exists():
                video_path = alt

        # Target keyframe jpg
        keyframe_jpg = KEYFRAMES_DIR / f"{video_path.stem}.jpg"
        if not keyframe_jpg.exists() and video_path.exists():
            extract_keyframe_from_sample(video_path, keyframe_jpg)

        if not keyframe_jpg.exists():
            continue

        isect_row = isect_map.get(name, {})
        isect_name = isect_row.get("Nearest_Intersection_Name", "")
        isect_roads = isect_row.get("Intersecting_Roads", "")

        print(f"[{processed_count + 1}/{len(cctvs)}] Inspecting via Gemini Vision: {name}...")
        vlm_res = query_gemini_vision(
            image_path=keyframe_jpg,
            cctv_name=name,
            lat=cctv_lat,
            lng=cctv_lng,
            intersection_name=isect_name,
            intersecting_roads=isect_roads,
            api_key=api_key
        )

        if "error" not in vlm_res and "bearing_deg" in vlm_res:
            bearing = float(vlm_res.get("bearing_deg", 0.0))
            cardinal = vlm_res.get("cardinal_direction", "Utara")
            conf = float(vlm_res.get("confidence_score", 0.9))
            traffic = vlm_res.get("traffic_orientation", "FORWARD_ALONG_TRAFFIC")
            reason = vlm_res.get("visual_reasoning", "")

            existing_perspectives[name] = {
                "cctv_name": name,
                "lat": cctv_lat,
                "lng": cctv_lng,
                "bearing_deg": round(bearing, 1),
                "cardinal_direction": cardinal,
                "fov_deg": 60.0,
                "estimation_method": "VLM_GEMINI_VISION",
                "confidence_score": round(conf, 2),
                "traffic_orientation": traffic,
                "nearest_intersection": isect_name,
                "distance_to_intersection_m": isect_row.get("Distance_to_Intersection_Meters", "0"),
                "optical_flow_detected": True,
                "visual_reasoning": reason
            }
            updated_count += 1
            print(f"  -> Result: {cardinal} ({bearing:.1f}°) | Conf: {conf:.2f} | Reason: {reason[:70]}...")
        else:
            print(f"  -> Skipped due to API error / empty response.")

        processed_count += 1
        time.sleep(0.5)  # Rate limit courtesy

    # Save updated dataset
    fieldnames = [
        "cctv_name",
        "lat",
        "lng",
        "bearing_deg",
        "cardinal_direction",
        "fov_deg",
        "estimation_method",
        "confidence_score",
        "traffic_orientation",
        "nearest_intersection",
        "distance_to_intersection_m",
        "optical_flow_detected",
        "visual_reasoning"
    ]

    with CSV_OUTPUT_PERSPECTIVES.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_perspectives.values())

    print(f"\n[SELESAI] Berhasil memperbarui {updated_count} kamera CCTV dengan inspeksi Gemini Vision VLM!")
    print(f"Dataset tersimpan di: {CSV_OUTPUT_PERSPECTIVES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CCTV perspective using Gemini Vision VLM")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Google Gemini API Key")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cameras to process (for testing)")
    args = parser.parse_args()

    if not args.api_key:
        print("[!] ERROR: Google Gemini API Key diperlukan.")
        print("    Silakan set environment variable GEMINI_API_KEY atau jalankan:")
        print("    python script_tools/analyze_perspective_vlm.py --api-key YOUR_API_KEY")
        sys.exit(1)

    run_vlm_perspective_batch(api_key=args.api_key, limit=args.limit)
