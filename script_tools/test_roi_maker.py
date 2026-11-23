#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated unit & integration tests for script_tools/roi_maker.py.
Verifies math, geometry, data persistence, and headless GUI logic.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from script_tools.roi_maker import (
    polygon_area_px,
    segments_intersect,
    polygon_is_valid,
    load_existing,
    save_json,
    list_videos,
    ModernRoiMakerApp,
    OUTPUT_JSON,
    VIDEO_DIR
)


class TestRoiMakerMath(unittest.TestCase):
    def test_polygon_area_square(self):
        # 100x100 square -> area = 10,000 px²
        pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertAlmostEqual(polygon_area_px(pts), 10000.0, places=1)

    def test_polygon_area_triangle(self):
        pts = [[0, 0], [100, 0], [0, 100]]
        self.assertAlmostEqual(polygon_area_px(pts), 5000.0, places=1)

    def test_segments_intersect(self):
        p1 = (0, 0)
        p2 = (10, 10)
        p3 = (0, 10)
        p4 = (10, 0)
        # Crossed lines should intersect
        self.assertTrue(segments_intersect(p1, p2, p3, p4))

        # Parallel lines should not intersect
        p5 = (0, 5)
        p6 = (10, 5)
        p7 = (0, 15)
        p8 = (10, 15)
        self.assertFalse(segments_intersect(p5, p6, p7, p8))

    def test_polygon_is_valid_valid_quad(self):
        pts = [[100, 100], [300, 100], [300, 300], [100, 300]]
        valid, msg = polygon_is_valid(pts)
        self.assertTrue(valid, msg)

    def test_polygon_is_valid_bowtie(self):
        # Bowtie self-intersecting polygon
        pts = [[100, 100], [300, 300], [300, 100], [100, 300]]
        valid, msg = polygon_is_valid(pts)
        self.assertFalse(valid)
        self.assertIn("bowtie", msg.lower())

    def test_polygon_is_valid_too_small(self):
        # Very tiny polygon (< 400 px²)
        pts = [[10, 10], [15, 10], [15, 15], [10, 15]]
        valid, msg = polygon_is_valid(pts)
        self.assertFalse(valid)
        self.assertIn("kecil", msg.lower())

    def test_polygon_is_valid_wrong_point_count(self):
        pts = [[10, 10], [15, 10], [15, 15]]
        valid, msg = polygon_is_valid(pts)
        self.assertFalse(valid)
        self.assertIn("tepat 4 titik", msg.lower())


class TestRoiMakerPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_json = os.path.join(self.test_dir, "test_rois.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_json(self):
        data = {
            "cctv_test_01": {
                "cctv_id": "cctv_test_01",
                "video_name": "cctv_test_01.mp4",
                "rois": [
                    {
                        "roi_idx": 0,
                        "name": "lane_1",
                        "polygon": [[100, 100], [200, 100], [200, 300], [100, 300]],
                        "length_m": 15.0,
                        "width_m": 3.5
                    }
                ],
                "perspective_deg": 45.0,
                "frame_width": 1280,
                "frame_height": 720
            }
        }
        # Monkeypatch target file for test
        import script_tools.roi_maker as rm
        old_output = rm.OUTPUT_JSON
        rm.OUTPUT_JSON = self.test_json
        try:
            rm.save_json(data)
            self.assertTrue(os.path.exists(self.test_json))

            loaded = rm.load_existing()
            self.assertIn("cctv_test_01", loaded)
            self.assertEqual(len(loaded["cctv_test_01"]["rois"]), 1)
            self.assertEqual(loaded["cctv_test_01"]["rois"][0]["name"], "lane_1")
            self.assertEqual(loaded["cctv_test_01"]["perspective_deg"], 45.0)

            # Test backup .bak creation on second save
            data["cctv_test_01"]["rois"].append({
                "roi_idx": 1,
                "name": "lane_2",
                "polygon": [[250, 100], [350, 100], [350, 300], [250, 300]],
                "length_m": 15.0,
                "width_m": 3.5
            })
            rm.save_json(data)
            self.assertTrue(os.path.exists(self.test_json + ".bak"))
        finally:
            rm.OUTPUT_JSON = old_output


class TestRoiMakerAppUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create App without showing on screen
        cls.app = ModernRoiMakerApp()
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()

    def test_app_initialization(self):
        self.assertIsNotNone(self.app.store)
        self.assertIsNotNone(self.app.video_items_container)
        self.assertIsNotNone(self.app.canvas)
        self.assertEqual(self.app.zoom, 1.0)
        self.assertEqual(len(self.app.points), 0)

    def test_video_list_and_filter(self):
        vids = list_videos()
        self.assertGreater(len(vids), 0, "Should find video samples in cctv_video_samples")

        # Test filter
        self.app.current_filter = "all"
        self.app.refresh_video_list()
        self.app.update()

        self.app.current_filter = "annotated"
        self.app.refresh_video_list()
        self.app.update()

        self.app.current_filter = "pending"
        self.app.refresh_video_list()
        self.app.update()

    def test_load_video_and_seek(self):
        vids = list_videos()
        first_vid = vids[0]
        self.app.load_video_by_name(first_vid)
        self.app.update()

        self.assertIsNotNone(self.app.frame)
        self.assertGreater(self.app.total_frames, 0)
        self.assertEqual(self.app.video_name, first_vid)

        # Test frame stepping
        self.app.step_frame(5)
        self.assertEqual(self.app.current_frame_idx, min(5, self.app.total_frames - 1))

        self.app.step_frame(-2)
        self.assertEqual(self.app.current_frame_idx, max(0, min(3, self.app.total_frames - 1)))

    def test_point_manipulation_and_drawing(self):
        vids = list_videos()
        self.app.load_video_by_name(vids[0])
        self.app.update()

        # Add 4 points
        self.app.points = [[100, 100], [300, 100], [300, 400], [100, 400]]
        self.app.update_roi_point_counters()
        self.assertEqual(len(self.app.points), 4)

        # Drag vertex 2
        self.app.drag_vertex_idx = 2
        self.app.points[2] = [320, 420]
        self.app.update_telemetry_metrics()
        self.assertEqual(self.app.points[2], [320, 420])
        self.app.drag_vertex_idx = None

        # Redraw
        self.app.redraw()
        self.app.update()

        # Undo
        self.app.undo_point()
        self.assertEqual(len(self.app.points), 3)

        # Reset
        self.app.reset_points()
        self.assertEqual(len(self.app.points), 0)

    def test_perspective_setting(self):
        self.app.set_perspective(135.0)
        self.assertEqual(self.app.entry.get("perspective_deg"), 135.0)
        self.app.redraw()
        self.app.update()

    def test_snapshot_rendering(self):
        vids = list_videos()
        self.app.load_video_by_name(vids[0])
        rendered = self.app._render_annotated_frame(self.app.frame.copy())
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, self.app.frame.shape)


if __name__ == "__main__":
    unittest.main()
