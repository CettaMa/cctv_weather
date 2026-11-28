#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Integration & Unit Test Suite for Bandung CCTV & Weather Platform.
Tests all new features:
1. Traffic Analyzer & Speed Estimation (Homography, Centroid Tracking, MKJI Level of Service).
2. Web Dashboard & API Endpoints (/api/rois, /api/traffic-telemetry, /api/corridors, /api/flood-risk, /api/stream-events).
3. In-memory HLS stream chunk caching.
"""

import os
import sys
import json
import re
import unittest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from script_tools.traffic_analyzer import (
    compute_bev_homography,
    transform_point,
    compute_level_of_service,
    analyze_video_traffic,
    VehicleTrack
)
import app as flask_app


class TestTrafficAnalyzerMath(unittest.TestCase):
    """Unit tests for Homography and Level of Service calculations."""

    def test_homography_and_transform(self):
        # 100x200 pixel rectangle mapped to 3.5m width x 15m length
        poly = [[0, 0], [100, 0], [100, 200], [0, 200]]
        length_m = 15.0
        width_m = 3.5

        H, H_inv = compute_bev_homography(poly, length_m, width_m)
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (3, 3))

        # Check corners transform exactly to metric plane
        p0_m = transform_point(H, (0, 0))
        p1_m = transform_point(H, (100, 0))
        p2_m = transform_point(H, (100, 200))
        p3_m = transform_point(H, (0, 200))

        self.assertAlmostEqual(p0_m[0], 0.0, places=2)
        self.assertAlmostEqual(p0_m[1], 0.0, places=2)
        self.assertAlmostEqual(p1_m[0], width_m, places=2)
        self.assertAlmostEqual(p1_m[1], 0.0, places=2)
        self.assertAlmostEqual(p2_m[0], width_m, places=2)
        self.assertAlmostEqual(p2_m[1], length_m, places=2)
        self.assertAlmostEqual(p3_m[0], 0.0, places=2)
        self.assertAlmostEqual(p3_m[1], length_m, places=2)

    def test_level_of_service_mapping(self):
        code_a, desc_a, col_a = compute_level_of_service(55.0, 10.0)
        self.assertEqual(code_a, "A")
        self.assertEqual(col_a, "#10B981")

        code_b, desc_b, col_b = compute_level_of_service(38.0, 15.0)
        self.assertEqual(code_b, "B")

        code_c, desc_c, col_c = compute_level_of_service(28.0, 20.0)
        self.assertEqual(code_c, "C")

        code_d, desc_d, col_d = compute_level_of_service(22.0, 25.0)
        self.assertEqual(code_d, "D")

        code_e, desc_e, col_e = compute_level_of_service(14.0, 30.0)
        self.assertEqual(code_e, "E")

        code_f, desc_f, col_f = compute_level_of_service(6.0, 40.0)
        self.assertEqual(code_f, "F")
        self.assertEqual(col_f, "#EF4444")

    def test_vehicle_track_speed_accumulation(self):
        vt = VehicleTrack(1, (50, 50), (1.0, 2.0), 0)
        # Move 10 meters in 1 second (30 frames at 30 fps) -> 10 m/s = 36 km/h
        vt.update((50, 100), (1.0, 12.0), 30, 30.0, 3000)
        self.assertEqual(len(vt.speeds_kmh), 1)
        self.assertAlmostEqual(vt.get_avg_speed(), 36.0, places=1)
        self.assertEqual(vt.vehicle_type, "mobil")


class TestFlaskPlatformAPI(unittest.TestCase):
    """Integration tests for all Flask endpoints in app.py."""

    @classmethod
    def setUpClass(cls):
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def test_index_dashboard(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"TENKOU-TK", res.data)
        self.assertIn(b"CCTV Wall Multi-Grid", res.data)
        self.assertIn(b"Pemantauan Titik Rawan Banjir", res.data)

    def test_buahbatu_forecast_page(self):
        res = self.client.get("/ramalan/buahbatu")
        self.assertEqual(res.status_code, 200)
        for element in (b"bbfRoadMap", b"bbfRainMap", b"bbfTraffic", b"bbfHeatAuto"):
            self.assertIn(element, res.data)
        self.assertIn(b'data-stream="/api/stream/HIKSVISION/Terbuahbatu.m3u8"', res.data)
        old = self.client.get("/?ramalan=buahbatu")
        self.assertEqual(old.status_code, 302)
        self.assertTrue(old.headers["Location"].endswith("/ramalan/buahbatu"))

    def test_about_page(self):
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        for logo in ("teknik-komputer", "dpri-ku", "brin", "atcs"):
            self.assertIn(f"/static/logos/{logo}.png".encode(), res.data)
            image = self.client.get(f"/static/logos/{logo}.png")
            self.assertEqual(image.status_code, 200)
            image.close()
        self.assertIn(b'href="/about"', self.client.get("/").data)

    def test_map_and_forecast_translations_are_complete(self):
        literal = r'"(?:[^"\\]|\\.)*"'
        pages = (
            ("/", "i18n_peta.js", [BASE_DIR / "templates" / "index.html"]),
            ("/ramalan/buahbatu", "i18n_ramalan.js",
             [BASE_DIR / "static" / "buahbatu_forecast.js", BASE_DIR / "static" / "buahbatu_live.js"]),
        )
        for url, dictionary, sources in pages:
            html = self.client.get(url).get_data(as_text=True)
            for lang in ("id", "ja", "en"):
                self.assertIn(f'data-lang="{lang}"', html)
            script = (BASE_DIR / "static" / dictionary).read_text(encoding="utf-8")
            block = script.split("I18n.add({", 1)[1].split("I18n.addValues(", 1)[0]
            entries = dict(re.findall(rf"^  (\w+): \[((?:\s*{literal},?)+)\s*\],", block, re.M))
            for key, body in entries.items():
                texts = re.findall(literal, body)
                self.assertEqual(len(texts), 3, f"{dictionary}: {key}")
                self.assertNotIn('""', texts, f"{dictionary}: {key}")
            self.assertEqual(len(entries), len(re.findall(r"^  \w+: \[", block, re.M)), dictionary)
            used = set(re.findall(r'data-i18n(?:-title|-aria|-placeholder)?="(\w+)"', html))
            for source in sources:
                used |= set(re.findall(r"""\bt\(["'](\w+)["']""", source.read_text(encoding="utf-8")))
            self.assertEqual(used - set(entries), set(), dictionary)

    def test_about_page_translations_are_complete(self):
        html = self.client.get("/about").get_data(as_text=True)
        used = set(re.findall(r'data-i18n="(\w+)"', html)) | {"title"}
        script = (BASE_DIR / "static" / "about.js").read_text(encoding="utf-8")
        for lang in ("en", "id", "ja"):
            self.assertIn(f'data-lang="{lang}"', html)
            block = re.search(rf"\n    {lang}: \{{(.*?)\n    \}},", script, re.S).group(1)
            keys = set(re.findall(r"^      (\w+):", block, re.M))
            self.assertEqual(keys, used, lang)

    def test_api_intersections(self):
        res = self.client.get("/api/intersections")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("cctvCount", data[0])
        self.assertIn("lat", data[0])

    def test_api_perspectives(self):
        res = self.client.get("/api/perspectives")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("bearing", data[0])
        self.assertIn("fov", data[0])

    def test_api_rois(self):
        res = self.client.get("/api/rois")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)

    def test_api_traffic_telemetry(self):
        res = self.client.get("/api/traffic-telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("telemetry", data)

    def test_api_corridors(self):
        res = self.client.get("/api/corridors")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 7, "Should have 7 pre-configured road corridors")
        self.assertEqual(data[0]["id"], "koridor_asia_afrika")
        self.assertIn("cameras", data[0])
        self.assertGreater(data[0]["cameraCount"], 0)

    def test_api_flood_risk(self):
        res = self.client.get("/api/flood-risk")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        hotspots = data.get("hotspots", [])
        self.assertEqual(len(hotspots), 5, "Should have 5 critical flood hotspots")
        self.assertEqual(hotspots[0]["id"], "flood_pasteur")
        self.assertIn("drainage", hotspots[0])
        self.assertIn("cctvs", hotspots[0])

    def test_api_status_and_refresh_state(self):
        res_status = self.client.get("/api/status?meta=1")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.get_json()
        self.assertIn("statuses", data_status)
        self.assertIn("state", data_status)

        res_state = self.client.get("/api/refresh-state")
        self.assertEqual(res_state.status_code, 200)
        data_state = res_state.get_json()
        self.assertIn("is_refreshing", data_state)

    def test_stream_caching_mechanism(self):
        # Test in-memory cache directly
        test_subpath = "test_camera/chunk_001.ts"
        dummy_chunk = b"\x47\x40\x00\x10" + b"\xFF" * 184

        with flask_app.STREAM_CACHE_LOCK:
            flask_app.STREAM_CACHE[test_subpath] = (dummy_chunk, flask_app.time.time())

        # Request chunk through proxy
        res = self.client.get(f"/api/stream/{test_subpath}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data, dummy_chunk)
        self.assertEqual(res.headers.get("X-Cache"), "HIT")
        self.assertEqual(res.headers.get("Content-Type"), "video/MP2T")

    def test_api_open_meteo_weather_grid(self):
        res = self.client.get("/api/weather/grid")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        grid_points = data.get("grid_points", [])
        self.assertEqual(len(grid_points), 12, "Should have 12 weather grid points")
        self.assertIn("temperature", grid_points[0])
        self.assertIn("precipitation", grid_points[0])
        self.assertIn("weather_icon", grid_points[0])

    def test_api_elevation_grid(self):
        res = self.client.get("/api/elevation/grid")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("rows"), 55, "DEM grid should span 55 latitude rows")
        self.assertEqual(data.get("cols"), 71, "DEM grid should span 71 longitude columns")
        elevations = data.get("elevations", [])
        self.assertEqual(len(elevations), 55 * 71)
        # Bandung basin floor sits near 600-700 m; the northern rim exceeds 1500 m.
        self.assertTrue(400 < data["min_elevation_m"] < 750)
        self.assertGreater(data["max_elevation_m"], 1500)
        # Row 28 / col 34 is the sample nearest Alun-Alun Bandung (-6.92, 107.62).
        self.assertTrue(600 <= elevations[28 * 71 + 34] <= 800)

    def test_api_temperature_grid(self):
        res = self.client.get("/api/weather/temperature-grid")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertTrue(data.get("is_placeholder"))
        self.assertEqual(data.get("rows"), 55)
        self.assertEqual(data.get("cols"), 71)
        times = data.get("times", [])
        frames = data.get("frames", [])
        self.assertEqual(len(times), 24, "Should provide 24 hourly forecast frames")
        self.assertEqual(len(frames), 24, "Should provide 24 hourly data frames")
        for frame in frames:
            self.assertEqual(len(frame), 55 * 71)

        # Lapse rate invariant: Lembang (row 7, col 34, elevation > 1200 m) must be
        # noticeably cooler than Alun-Alun Bandung (row 28, col 34, elevation ~ 680 m).
        lembang_t0 = frames[0][7 * 71 + 34]
        alun_alun_t0 = frames[0][28 * 71 + 34]
        self.assertIsNotNone(lembang_t0)
        self.assertIsNotNone(alun_alun_t0)
        self.assertLess(lembang_t0, alun_alun_t0)

        # Diurnal cycle: 14:00 peak frame must be warmer than 02:00 trough frame
        idx_14 = next(i for i, t in enumerate(times) if t[11:13] == "14")
        idx_02 = next(i for i, t in enumerate(times) if t[11:13] == "02")
        self.assertGreater(frames[idx_14][28 * 71 + 34], frames[idx_02][28 * 71 + 34])

        # Global bounds consistency
        min_t = data.get("min_temperature_c")
        max_t = data.get("max_temperature_c")
        self.assertIsNotNone(min_t)
        self.assertIsNotNone(max_t)
        self.assertLess(min_t, max_t)
        for frame in frames:
            for val in frame:
                if val is not None:
                    self.assertTrue(min_t <= val <= max_t)

    def test_api_weather_classify_frame_rain_and_sunny(self):
        import cv2
        for cls_name in ("rain", "sunny"):
            clip_paths = list((BASE_DIR / "weather" / "kordon" / "dataset" / "video").glob(f"*_{cls_name}.mp4"))
            if not clip_paths:
                continue
            cap = cv2.VideoCapture(str(clip_paths[0]))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                continue
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.assertTrue(ok_enc)
            res = self.client.post("/api/weather/classify-frame", data=buf.tobytes(), content_type="image/jpeg")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get("status"), "success")
            self.assertEqual(data.get("label"), cls_name)
            self.assertIsNotNone(data.get("p_rain"))
            self.assertTrue(0.0 <= data["p_rain"] <= 1.0)
            self.assertGreaterEqual(data["confidence"], 0.5)
            if cls_name == "rain":
                self.assertGreater(data["p_rain"], 0.5)
            else:
                self.assertLess(data["p_rain"], 0.5)

    def test_api_weather_classify_frame_low_light(self):
        import cv2
        dark_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dark_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res = self.client.post("/api/weather/classify-frame", data=buf.tobytes(), content_type="image/jpeg")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("label"), "unknown")
        self.assertIsNone(data.get("p_rain"))
        self.assertLess(data.get("brightness", 999), 60.0)

    def test_api_weather_classify_frame_bad_input(self):
        # Empty payload
        res_empty = self.client.post("/api/weather/classify-frame", data=b"", content_type="image/jpeg")
        self.assertEqual(res_empty.status_code, 400)
        self.assertEqual(res_empty.get_json().get("status"), "error")

        # Corrupted payload
        res_corrupt = self.client.post("/api/weather/classify-frame", data=b"not-a-valid-jpeg-bytes", content_type="image/jpeg")
        self.assertEqual(res_corrupt.status_code, 400)
        self.assertEqual(res_corrupt.get_json().get("status"), "error")

    def test_api_open_meteo_wind_field(self):
        res = self.client.get("/api/weather/wind-field")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        wind_vectors = data.get("wind_vectors", [])
        self.assertEqual(len(wind_vectors), 12)
        self.assertIn("speed", wind_vectors[0])
        self.assertIn("deg", wind_vectors[0])

    def test_api_open_meteo_hydrology(self):
        res = self.client.get("/api/weather/hydrology")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        stations = data.get("stations", [])
        self.assertEqual(len(stations), 5, "Should have 5 major river basin stations")
        self.assertIn("current_discharge_m3s", stations[0])
        self.assertIn("status", stations[0])


if __name__ == "__main__":
    unittest.main()
