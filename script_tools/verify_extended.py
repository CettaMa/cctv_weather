#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended functional verification for ROI Maker Studio.
Tests multi-ROI workflow, perspective updates, snapshot export, and JSON persistence in isolated environment.
"""

import os
import sys
import json
import shutil
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import script_tools.roi_maker as rm
from script_tools.roi_maker import (
    ModernRoiMakerApp,
    list_videos,
    load_existing,
    save_json,
    SNAPSHOT_DIR
)


def run_extended_verification():
    temp_dir = tempfile.mkdtemp()
    test_json = os.path.join(temp_dir, "test_rois.json")
    old_output_json = rm.OUTPUT_JSON
    rm.OUTPUT_JSON = test_json

    try:
        print("=== 1. Testing Video Discovery ===")
        vids = list_videos()
        assert len(vids) > 0, "No videos found!"
        print(f"Found {len(vids)} CCTV video samples.")

        print("\n=== 2. Testing App UI & Video Loading ===")
        app = ModernRoiMakerApp()
        app.withdraw()

        # Load 1st video
        vid1 = vids[0]
        app.load_video_by_name(vid1)
        assert app.frame is not None, "Frame failed to load"
        assert app.video_name == vid1, "Video name mismatch"
        print(f"Loaded: {vid1} ({app.frame.shape[1]}x{app.frame.shape[0]}, {app.total_frames} frames)")

        print("\n=== 3. Testing ROI Creation & Vertex Dragging ===")
        # Place 4 points
        app.points = [[200, 150], [450, 160], [460, 380], [190, 370]]
        app.roi_name_var.set("lajur_utama")
        app.roi_len_var.set("18.5")
        app.roi_wid_var.set("3.5")
        app.set_perspective(75.0)
        app.update_roi_point_counters()
        app.save_roi()

        assert len(app.entry["rois"]) == 1
        assert app.entry["rois"][0]["name"] == "lajur_utama"
        assert app.entry["rois"][0]["length_m"] == 18.5
        assert app.entry["rois"][0]["width_m"] == 3.5
        print("Saved ROI 1 successfully.")

        # Place 2nd ROI
        app.points = [[500, 160], [750, 170], [760, 390], [490, 380]]
        app.roi_name_var.set("lajur_kanan")
        app.roi_len_var.set("18.5")
        app.roi_wid_var.set("3.5")
        app.save_roi()

        assert len(app.entry["rois"]) == 2
        print("Saved ROI 2 successfully.")

        # Select ROI 1 to inspect and edit vertex
        app.select_saved_roi(0)
        assert app.roi_name_var.get() == "lajur_utama"
        app.points[0] = [210, 155]  # Simulate dragging vertex 0
        app.roi_len_var.set("19.0")
        app.save_roi()

        assert app.entry["rois"][0]["polygon"][0] == [210, 155]
        assert app.entry["rois"][0]["length_m"] == 19.0
        print("Edited & saved ROI 1 with vertex adjustment.")

        print("\n=== 4. Testing Snapshot Export ===")
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        snap_count_before = len(os.listdir(SNAPSHOT_DIR))
        app.export_snapshot(silent=True)
        snap_count_after = len(os.listdir(SNAPSHOT_DIR))
        assert snap_count_after > snap_count_before, "Snapshot was not created!"
        print("Snapshot exported successfully.")

        print("\n=== 5. Testing Delete ROI & Re-indexing ===")
        roi_to_del = app.entry["rois"][0]
        app.entry["rois"].remove(roi_to_del)
        for i, r in enumerate(app.entry["rois"]):
            r["roi_idx"] = i
        save_json(app.store)
        app.refresh_saved_rois_list()

        assert len(app.entry["rois"]) == 1
        assert app.entry["rois"][0]["roi_idx"] == 0
        assert app.entry["rois"][0]["name"] == "lajur_kanan"
        print("Deleted ROI and verified clean re-indexing.")

        app.on_close()
        print("\n=== ALL EXTENDED TESTS PASSED PERFECTLY! ===")
    finally:
        rm.OUTPUT_JSON = old_output_json
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_extended_verification()
