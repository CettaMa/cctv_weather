#!/usr/bin/env python3
"""Merge roi_annotations.json files from paket_1..paket_4 into one master JSON."""

import glob
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "roi_packages")
OUT_FILE = os.path.join(BASE_DIR, "outputs", "all_roi_annotations.json")


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    pattern = os.path.join(PACKAGES_DIR, "paket_*", "outputs", "roi_annotations.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Tidak ditemukan file anotasi di: {pattern}")
        return

    merged = {
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [],
        "total_cameras": 0,
        "cameras": {},
    }

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged["sources"].append(
            {
                "file": path,
                "package_name": data.get("package_name", ""),
                "annotator": data.get("annotator", ""),
                "updated_at": data.get("updated_at", ""),
            }
        )
        for cam_key, cam_val in data.get("cameras", {}).items():
            cam_copy = dict(cam_val)
            cam_copy["annotator"] = data.get("annotator", "")
            cam_copy["package_name"] = data.get("package_name", "")
            merged["cameras"][cam_key] = cam_copy

    merged["total_cameras"] = len(merged["cameras"])
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(
        f"Berhasil menggabungkan {len(files)} file ({merged['total_cameras']} CCTV) -> {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
