#!/usr/bin/env python3
"""Check all 26 RTSP CCTV streams from Excel and capture ~5s MP4 video samples."""

import argparse
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
EXCEL_PATH = os.path.join(ROOT_DIR, "RTSP VID - Telkom University.xlsx")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RTSP_HOST = "atcs-dishub.bandung.go.id:9104"


def rtsp_url(channel: str) -> str:
    """URL RTSP dengan kredensial dari environment atau .env di root proyek (tidak disimpan di kode)."""
    env = {}
    dotenv = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(dotenv):
        with open(dotenv, encoding="utf-8") as handle:
            for line in handle:
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip().strip("\"'")
    env.update(os.environ)
    user, password = env.get("RTSP_USER", ""), env.get("RTSP_PASS", "")
    if not user or not password:
        raise SystemExit("Isi RTSP_USER dan RTSP_PASS di .env (lihat .env.example).")
    auth = f"{quote(user, safe='')}:{quote(password, safe='')}"
    return f"rtsp://{auth}@{RTSP_HOST}/Streaming/Channels/{channel}"


WARMUP_FRAMES = 25      # skip initial P-frame artifacts before keyframe settles
RECORD_SECONDS = 5.0    # record 5 seconds of clean video
MAX_TOTAL_SEC = 35.0    # safety timeout per stream


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_")
    return cleaned.lower()


def load_cctv_list(excel_path: str) -> list[dict]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(excel_path, "r") as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall("m:si", ns):
                parts = [t.text or "" for t in si.findall(".//m:t", ns)]
                shared_strings.append("".join(parts))

        root_ws = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = root_ws.findall(".//m:sheetData/m:row", ns)

    items = []
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        cells = {}
        for c in row.findall("m:c", ns):
            ref = c.attrib.get("r", "")
            col = re.match(r"[A-Z]+", ref).group(0) if ref else ""
            ctype = c.attrib.get("t", "")
            v = c.find("m:v", ns)
            val = v.text if v is not None else None
            if val is not None and ctype == "s":
                val = shared_strings[int(val)]
            cells[col] = val

        channel, name = cells.get("A"), cells.get("B")
        if channel is None or name is None:
            continue
        channel_str = str(channel).strip()
        name_str = str(name).strip()
        items.append(
            {
                "channel": channel_str,
                "name": name_str,
                "rtsp_url": rtsp_url(channel_str),
                "filename": f"{channel_str}_{sanitize_filename(name_str)}.mp4",
            }
        )
    return items


def capture_one(item: dict, out_dir: str, transport: str = "tcp") -> dict:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|stimeout;10000000|rw_timeout;10000000"
        "|probesize;5000000|analyzeduration;5000000"
    )
    url = item["rtsp_url"]
    t0 = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    result = {
        "channel": item["channel"],
        "name": item["name"],
        "rtsp_url": url,
        "filename": item["filename"],
        "status": "FAILED",
        "width": 0,
        "height": 0,
        "frames": 0,
        "fps": 0.0,
        "error": "",
        "elapsed_sec": 0.0,
    }
    writer = None
    out_path = os.path.join(out_dir, item["filename"])
    try:
        if not cap.isOpened():
            result["error"] = "Cannot open RTSP stream"
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 60:
            fps = 25.0

        # Read warmup frames so decoder reaches a clean I-frame
        first_valid = None
        for i in range(WARMUP_FRAMES + 10):
            if time.time() - t0 > MAX_TOTAL_SEC:
                break
            ret, f = cap.read()
            if ret and f is not None and f.size > 0:
                first_valid = f
                if i >= WARMUP_FRAMES:
                    break

        if first_valid is None:
            result["error"] = "Stream opened but failed to read frame"
            return result

        h, w = first_valid.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            result["error"] = "Failed to open VideoWriter"
            return result

        writer.write(first_valid)
        written = 1
        target_frames = max(25, int(round(fps * RECORD_SECONDS)))

        while written < target_frames and (time.time() - t0) < MAX_TOTAL_SEC:
            ret, f = cap.read()
            if not ret or f is None or f.size == 0:
                break
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
            written += 1

        writer.release()
        writer = None

        if written < 5 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            result["error"] = f"Too few frames recorded ({written})"
            return result

        result["status"] = "OK"
        result["width"] = int(w)
        result["height"] = int(h)
        result["frames"] = int(written)
        result["fps"] = round(float(fps), 2)
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        result["elapsed_sec"] = round(time.time() - t0, 2)


def capture_with_fallback(item: dict, out_dir: str) -> dict:
    res = capture_one(item, out_dir, transport="tcp")
    if res["status"] != "OK" and "Cannot open" not in res["error"]:
        res_udp = capture_one(item, out_dir, transport="udp")
        if res_udp["status"] == "OK":
            return res_udp
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry channels that failed in the previous report.",
    )
    args = parser.parse_args()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    report_json = os.path.join(SAMPLES_DIR, "rtsp_check_report.json")
    items = load_cctv_list(EXCEL_PATH)
    existing_map: dict[str, dict] = {}

    if args.retry_failed and os.path.exists(report_json):
        with open(report_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
        existing_map = {str(r["channel"]): r for r in prev}
        to_process = [
            it
            for it in items
            if existing_map.get(it["channel"], {}).get("status") != "OK"
        ]
        print(f"Retrying {len(to_process)} failed CCTV channels sequentially...")
        results = [
            existing_map[it["channel"]]
            for it in items
            if existing_map.get(it["channel"], {}).get("status") == "OK"
        ]
        for it in to_process:
            res = capture_with_fallback(it, SAMPLES_DIR)
            results.append(res)
            tag = "OK" if res["status"] == "OK" else "FAIL"
            detail = (
                f"{res['width']}x{res['height']} ({res['frames']}f @ {res['fps']}fps)"
                if res["status"] == "OK"
                else res["error"]
            )
            print(
                f"[{tag}] {res['channel']:>4} | {res['name']:<24} | "
                f"{detail} ({res['elapsed_sec']}s)"
            )
    else:
        for old_f in os.listdir(SAMPLES_DIR):
            if old_f.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(os.path.join(SAMPLES_DIR, old_f))
        print(f"Loaded {len(items)} CCTV channels from Excel.")
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_map = {
                pool.submit(capture_with_fallback, item, SAMPLES_DIR): item
                for item in items
            }
            for fut in as_completed(future_map):
                res = fut.result()
                results.append(res)
                tag = "OK" if res["status"] == "OK" else "FAIL"
                detail = (
                    f"{res['width']}x{res['height']} ({res['frames']}f @ {res['fps']}fps)"
                    if res["status"] == "OK"
                    else res["error"]
                )
                print(
                    f"[{tag}] {res['channel']:>4} | {res['name']:<24} | "
                    f"{detail} ({res['elapsed_sec']}s)"
                )

    results.sort(key=lambda r: int(r["channel"]))

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_csv = os.path.join(SAMPLES_DIR, "rtsp_check_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "name",
                "status",
                "width",
                "height",
                "frames",
                "fps",
                "filename",
                "elapsed_sec",
                "error",
                "rtsp_url",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - ok_count
    print(f"\nSummary: {ok_count} OK, {fail_count} FAILED (total {len(results)})")
    print(f"Saved report to {report_json} and {report_csv}")


if __name__ == "__main__":
    main()
