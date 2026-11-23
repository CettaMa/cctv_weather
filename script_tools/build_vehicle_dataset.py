#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle detection dataset pipeline: sample frames from 24/7 ATCS CCTV recordings,
pre-label zero-shot with Grounding DINO on CUDA, and export to Ultralytics YOLO format.

    python -m script_tools.build_vehicle_dataset sample [--root PATH] [--total 4000] [--pool-factor 3] [--workers 4] [--dry-run]
    python -m script_tools.build_vehicle_dataset prelabel [--root PATH] [--model IDEA-Research/grounding-dino-base] [--floor 0.20]
    python -m script_tools.build_vehicle_dataset export [--root PATH] [--threshold 0.30]

Dependencies (torch, torchvision, transformers, ultralytics) are imported lazily inside
the subcommands that require them.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from datetime import date, datetime, time, timedelta
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Hard source protection rules
# --------------------------------------------------------------------------- #

SOURCE_DRIVES = {"D:", "E:", "F:", "G:"}


def _normalize_path(p: Path) -> Path:
    s = str(p.resolve())
    while s.startswith("\\\\?\\"):
        s = s[4:]
    return Path(s)


def _get_drive(p: Path) -> str:
    norm = _normalize_path(p)
    return norm.drive.upper().rstrip(":") + ":"


def _resolve_root(path_str: str | Path) -> Path:
    p = _normalize_path(Path(path_str))
    if _get_drive(p) in SOURCE_DRIVES:
        sys.exit(f"Error: root must not be on a source drive (D:/E:/F:/G:): {p}")
    return p


def _assert_under_root(path: Path, root: Path) -> Path:
    p = _normalize_path(path)
    r = _normalize_path(root)
    if not p.is_relative_to(r) or _get_drive(p) in SOURCE_DRIVES:
        sys.exit(f"refusing to operate outside root or on source drive: {p}")
    return p


def _safe_remove(path: Path, root: Path) -> None:
    p = _assert_under_root(path, root)
    if p.is_file():
        p.unlink()


def _safe_rmtree(path: Path, root: Path) -> None:
    p = _assert_under_root(path, root)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


# --------------------------------------------------------------------------- #
# Camera registry & constants
# --------------------------------------------------------------------------- #

CAMERAS: dict[str, Path] = {
    "mohtoha_barat": Path("D:/SP Moh Toha/Barat"),
    "mohtoha_selatan": Path("D:/SP Moh Toha/Selatan"),
    "mohtoha_timur": Path("D:/SP Moh Toha/Timur"),
    "mohtoha_utara": Path("D:/SP Moh Toha/Utara"),
    "samsat_barat": Path("E:/SP Samsat/Barat"),
    "samsat_utara": Path("E:/SP Samsat/Utara"),
    "buahbatu_barat": Path("F:/SP Buahbatu/Barat"),
    "buahbatu_timur": Path("F:/SP Buahbatu/Timur"),
    "buahbatu_utara": Path("G:/SP Buahbatu/Utara"),
    "gedebage_barat": Path("F:/SP Gedebage/Barat"),
    "gedebage_selatan": Path("F:/SP Gedebage/Selatan"),
    "batununggal_barat": Path("G:/SP Batununggal/Barat"),
    "batununggal_timur": Path("G:/SP Batununggal/Timur"),
}

CLASSES: tuple[str, ...] = ("car", "motorcycle", "truck", "bus")
FILE_RE = re.compile(r"_(\d{14})_(\d{14})_(\d+)_4G\.mp4$")

DAY_START = time(6, 30, 0)
DAY_END = time(17, 0, 0)
NIGHT_START = time(18, 45, 0)
NIGHT_END = time(5, 0, 0)

DEFAULT_ROOT = BASE_DIR / "datasets" / "vehicle_det"


# --------------------------------------------------------------------------- #
# Candidate intervals & sampling logic
# --------------------------------------------------------------------------- #

def _calculate_quotas(total: int) -> dict[str, int]:
    per_period = total // 2
    cam_keys = list(CAMERAS.keys())
    base, rem = divmod(per_period, len(cam_keys))
    quotas = {}
    for i, cam_id in enumerate(cam_keys):
        quotas[cam_id] = base + (1 if i < rem else 0)
    return quotas

def _probe_duration(vpath: Path) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(vpath)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(p.stdout)
        dur = float(data["format"]["duration"])
        return dur if dur > 0 else None
    except Exception:
        return None


def _get_file_durations(folders: list[Path], cache_file: Path | None = None) -> dict[str, float]:
    durations: dict[str, float] = {}
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                durations = json.load(f)
        except Exception:
            durations = {}

    all_files: list[Path] = []
    for fld in folders:
        if fld.exists():
            for p in fld.glob("*.mp4"):
                if str(p) not in durations:
                    all_files.append(p)

    if all_files:
        print(f"Probing actual video durations for {len(all_files)} files...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = ex.map(_probe_duration, all_files)
            for p, dur in zip(all_files, results):
                if dur is not None:
                    durations[str(p)] = dur
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(durations, f)

    return durations


def _get_allowed_intervals(
    folder: Path, period: str, durations: dict[str, float] | None = None
) -> list[tuple[datetime, datetime, float, Path, datetime, datetime]]:
    """
    Find all allowed time intervals for a camera folder and period (day or night).
    Each returned element is:
      (interval_start, interval_end, duration_seconds, video_path, file_start, file_end)
    """
    if not folder.exists():
        return []

    intervals: list[tuple[datetime, datetime, float, Path, datetime, datetime]] = []
    video_files = sorted(folder.glob("*.mp4"))

    for vpath in video_files:
        m = FILE_RE.search(vpath.name)
        if not m:
            continue
        try:
            f_start = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
            f_end = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            continue

        if durations is not None:
            dur = durations.get(str(vpath)) or durations.get(str(Path(vpath).resolve()))
            if dur is not None and dur > 0:
                f_end = min(f_end, f_start + timedelta(seconds=dur))
            elif dur is None:
                continue
        valid_s = f_start + timedelta(seconds=10)
        valid_e = f_end - timedelta(seconds=10)
        if valid_s >= valid_e:
            continue

        d_curr = valid_s.date()
        d_end = valid_e.date()

        while d_curr <= d_end:
            windows: list[tuple[datetime, datetime]] = []
            if period == "day":
                w_s = datetime.combine(d_curr, DAY_START)
                w_e = datetime.combine(d_curr, DAY_END)
                windows.append((w_s, w_e))
            elif period == "night":
                w_m_s = datetime.combine(d_curr, time(0, 0, 0))
                w_m_e = datetime.combine(d_curr, NIGHT_END)
                windows.append((w_m_s, w_m_e))
                w_e_s = datetime.combine(d_curr, NIGHT_START)
                w_e_e = datetime.combine(d_curr + timedelta(days=1), time(0, 0, 0))
                windows.append((w_e_s, w_e_e))

            for win_s, win_e in windows:
                ov_s = max(valid_s, win_s)
                ov_e = min(valid_e, win_e)
                if ov_s < ov_e:
                    dur = (ov_e - ov_s).total_seconds()
                    if dur > 0:
                        intervals.append((ov_s, ov_e, dur, vpath, f_start, f_end))

            d_curr += timedelta(days=1)

    return intervals


def _sample_candidates_for_cam(
    cam_id: str,
    folder: Path,
    period: str,
    quota: int,
    pool_factor: int,
    durations: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Generate stratified candidates for one camera and period.
    Returns (candidate_records, total_duration_seconds).
    """
    intervals = _get_allowed_intervals(folder, period, durations)
    if not intervals:
        return [], 0.0

    cum_lengths: list[float] = [0.0]
    for it in intervals:
        cum_lengths.append(cum_lengths[-1] + it[2])
    total_l = cum_lengths[-1]

    N = pool_factor * quota
    rng = random.Random(f"42-{cam_id}-{period}")
    candidates: list[dict[str, Any]] = []

    for k in range(N):
        u = (k + rng.random()) * total_l / N
        idx = bisect.bisect_right(cum_lengths, u) - 1
        idx = max(0, min(idx, len(intervals) - 1))
        delta = u - cum_lengths[idx]
        delta = max(0.0, min(delta, intervals[idx][2]))

        ts = intervals[idx][0] + timedelta(seconds=delta)
        f_start = intervals[idx][4]
        f_end = intervals[idx][5]
        vpath = intervals[idx][3]
        offset_s = (ts - f_start).total_seconds()

        cid = f"{cam_id}_{ts:%Y%m%d_%H%M%S}_{period}"
        s_date = ts.date() - timedelta(days=1) if ts.time() < time(5, 0) else ts.date()
        image_rel = f"candidates/images/{cam_id}/{cid}.jpg"

        candidates.append({
            "id": cid,
            "cam_id": cam_id,
            "period": period,
            "timestamp": ts.isoformat(timespec="seconds"),
            "session_date": s_date.isoformat(),
            "video": str(vpath),
            "file_start": f_start,
            "file_end": f_end,
            "offset_s": offset_s,
            "image": image_rel,
            "status": "pending",
        })

    return candidates, total_l


# --------------------------------------------------------------------------- #
# Frame extraction worker
# --------------------------------------------------------------------------- #

def _extract_one_frame(cand: dict[str, Any], root: Path) -> dict[str, Any]:
    image_path = _assert_under_root(root / cand["image"], root)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip extraction if non-empty image already exists
    if not (image_path.exists() and image_path.stat().st_size > 0):
        video_path = Path(cand["video"])
        offset_s = float(cand["offset_s"])

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-skip_frame", "nokey",
            "-ss", f"{offset_s:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            "-y", str(image_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0 or not image_path.exists() or image_path.stat().st_size == 0:
            retry_offset = offset_s + 30.0
            max_offset = (cand["file_end"] - cand["file_start"]).total_seconds() - 10.0
            if retry_offset <= max_offset:
                cmd_retry = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-skip_frame", "nokey",
                    "-ss", f"{retry_offset:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    "-y", str(image_path),
                ]
                subprocess.run(cmd_retry, capture_output=True, text=True)

    # Validate output
    if not image_path.exists() or image_path.stat().st_size == 0:
        cand["status"] = "decode_fail"
        return cand

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        cand["status"] = "decode_fail"
    elif float(img.std()) < 8.0:
        cand["status"] = "blank"
        _safe_remove(image_path, root)
    else:
        cand["status"] = "ok"

    return cand


# --------------------------------------------------------------------------- #
# Subcommand: sample
# --------------------------------------------------------------------------- #

def cmd_sample(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    total = int(args.total)
    pool_factor = int(args.pool_factor)
    workers = int(args.workers)
    dry_run = bool(args.dry_run)

    quotas = _calculate_quotas(total)
    print(f"Dataset root: {root}")
    print(f"Total target frames: {total} (Day: {total // 2}, Night: {total // 2})")
    dur_cache = (root / "candidates" / "file_durations.json") if not dry_run else None
    durations = _get_file_durations(list(CAMERAS.values()), dur_cache)

    all_candidates: list[dict[str, Any]] = []
    dry_rows: list[dict[str, Any]] = []

    for cam_id, folder in CAMERAS.items():
        q = quotas[cam_id]
        for period in ("day", "night"):
            cands, total_l = _sample_candidates_for_cam(cam_id, folder, period, q, pool_factor, durations)
            all_candidates.extend(cands)
            dry_rows.append({
                "cam_id": cam_id,
                "period": period,
                "quota": q,
                "N": len(cands),
                "L_hours": total_l / 3600.0,
                "first_two": cands[:2],
            })

    if dry_run:
        print("\n=== DRY RUN SUMMARY ===")
        print(f"{'Camera':<20} {'Period':<6} {'Quota':<6} {'N':<6} {'L (hours)':<10} {'First 2 candidates'}")
        print("-" * 80)
        for row in dry_rows:
            c1_str = ""
            if row["first_two"]:
                c1 = row["first_two"][0]
                c1_str = f"{c1['id']} ({Path(c1['video']).name}, offset {c1['offset_s']:.1f}s)"
            print(f"{row['cam_id']:<20} {row['period']:<6} {row['quota']:<6} {row['N']:<6} {row['L_hours']:<10.1f} {c1_str}")
            if len(row["first_two"]) > 1:
                c2 = row["first_two"][1]
                print(f"{'':<20} {'':<6} {'':<6} {'':<6} {'':<10} {c2['id']} ({Path(c2['video']).name}, offset {c2['offset_s']:.1f}s)")
        print("\nDry run completed. No files were written.")
        return 0

    cand_dir = root / "candidates"
    cand_img_dir = cand_dir / "images"
    cand_img_dir.mkdir(parents=True, exist_ok=True)

    config_path = cand_dir / "config.json"
    config_data = {
        "total": total,
        "pool_factor": pool_factor,
        "quotas": quotas,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"\nExtracting {len(all_candidates)} candidates using {workers} workers...")
    from concurrent.futures import ThreadPoolExecutor

    extracted: list[dict[str, Any]] = []
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for cand_res in executor.map(lambda c: _extract_one_frame(c, root), all_candidates):
            extracted.append(cand_res)
            count += 1
            if count % 200 == 0 or count == len(all_candidates):
                print(f"Extracted {count}/{len(all_candidates)} candidates...")

    manifest_path = cand_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cam_id", "period", "timestamp", "session_date", "video", "offset_s", "image", "status"])
        for c in extracted:
            writer.writerow([
                c["id"],
                c["cam_id"],
                c["period"],
                c["timestamp"],
                c["session_date"],
                c["video"],
                f"{float(c['offset_s']):.3f}",
                c["image"],
                c["status"],
            ])

    status_counts: dict[str, int] = {}
    for c in extracted:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    print("\nExtraction complete. Manifest status summary:")
    for st, cnt in sorted(status_counts.items()):
        print(f"  {st}: {cnt}")
    print(f"Manifest written to: {manifest_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: prelabel
# --------------------------------------------------------------------------- #

def cmd_prelabel(args: argparse.Namespace) -> int:
    import time as time_mod
    root = _resolve_root(args.root)
    model_id = str(args.model)
    floor = float(args.floor)

    try:
        import torch
        import torchvision
    except ImportError:
        sys.exit("Error: PyTorch and torchvision are required for prelabel. Please install CUDA torch.")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA is not available. Grounding DINO prelabeling requires GPU.")

    manifest_path = root / "candidates" / "manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"Error: manifest not found at {manifest_path}. Run sample first.")

    rows: list[dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] == "ok":
                rows.append(r)

    print(f"Found {len(rows)} 'ok' candidates in manifest.")

    detections_path = root / "candidates" / "detections.jsonl"
    existing_ids: set[str] = set()
    if detections_path.exists():
        with open(detections_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        existing_ids.add(d["id"])
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: found {len(existing_ids)} existing detections in detections.jsonl.")

    to_process = [r for r in rows if r["id"] not in existing_ids]
    print(f"{len(to_process)} images left to process.")
    if not to_process:
        print("All candidates already pre-labeled.")
        return 0

    print(f"Loading Grounding DINO model '{model_id}' onto CUDA...")
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    PROMPT = "car. motorcycle. truck. bus."
    inputs_text = processor.tokenizer(PROMPT, return_offsets_mapping=True)

    offsets: list[tuple[int, int]] = []
    if "offset_mapping" in inputs_text:
        om = inputs_text["offset_mapping"]
        if hasattr(om, "tolist"):
            om = om.tolist()
        if isinstance(om[0], list) and isinstance(om[0][0], (list, tuple)):
            offsets = [tuple(x) for x in om[0]]
        else:
            offsets = [tuple(x) for x in om]
    elif hasattr(inputs_text, "encodings") and inputs_text.encodings:
        offsets = [tuple(x) for x in inputs_text.encodings[0].offsets]

    tok_idx: dict[int, list[int]] = {}
    for k, name in enumerate(CLASSES):
        c_s = PROMPT.index(name)
        c_e = c_s + len(name)
        indices = [
            idx for idx, (s, e) in enumerate(offsets)
            if (s, e) != (0, 0) and max(s, c_s) < min(e, c_e)
        ]
        assert len(indices) >= 1, f"Class {name} must have >= 1 token in prompt"
        tok_idx[k] = indices

    print(f"Token indices mapped: { {CLASSES[k]: tok_idx[k] for k in tok_idx} }")

    tok_tensors = {k: torch.tensor(tok_idx[k], device=device, dtype=torch.long) for k in tok_idx}
    text_inputs = processor.tokenizer(PROMPT, return_tensors="pt").to(device)
    t0 = time_mod.perf_counter()
    count = 0
    with open(detections_path, "a", encoding="utf-8") as out_f:
        with torch.inference_mode():
            for r in to_process:
                img_path = root / r["image"]
                if not img_path.exists():
                    continue

                pil_img = Image.open(img_path).convert("RGB")
                W, H = pil_img.size

                image_inputs = processor.image_processor(images=pil_img, return_tensors="pt").to(device)
                inputs = {**image_inputs, **text_inputs}
                outputs = model(**inputs)

                probs = outputs.logits[0].sigmoid()  # (num_queries, 256)
                cls_scores = torch.zeros((probs.shape[0], len(CLASSES)), device=device, dtype=torch.float32)
                for k in range(len(CLASSES)):
                    cls_scores[:, k] = probs[:, tok_tensors[k]].max(dim=1).values

                scores, clses = cls_scores.max(dim=1)
                mask = scores >= floor

                kept_boxes_list: list[list[float]] = []
                if mask.any():
                    boxes_norm = outputs.pred_boxes[0][mask]
                    scores_kept = scores[mask]
                    clses_kept = clses[mask]

                    cx = boxes_norm[:, 0] * W
                    cy = boxes_norm[:, 1] * H
                    bw = boxes_norm[:, 2] * W
                    bh = boxes_norm[:, 3] * H
                    x1 = (cx - bw / 2.0).clamp(0.0, float(W))
                    y1 = (cy - bh / 2.0).clamp(0.0, float(H))
                    x2 = (cx + bw / 2.0).clamp(0.0, float(W))
                    y2 = (cy + bh / 2.0).clamp(0.0, float(H))
                    box_w = x2 - x1
                    box_h = y2 - y1
                    area = box_w * box_h

                    valid = (box_w >= 6.0) & (box_h >= 6.0) & (area <= 0.4 * W * H)
                    if valid.any():
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)[valid]
                        scores_val = scores_kept[valid]
                        clses_val = clses_kept[valid]

                        nms_idx = torchvision.ops.nms(boxes_xyxy, scores_val, 0.7)
                        for idx in nms_idx:
                            b = boxes_xyxy[idx]
                            c = int(clses_val[idx])
                            s = float(scores_val[idx])
                            kept_boxes_list.append([
                                round(float(b[0]), 1),
                                round(float(b[1]), 1),
                                round(float(b[2]), 1),
                                round(float(b[3]), 1),
                                c,
                                round(s, 4),
                            ])

                record = {
                    "id": r["id"],
                    "w": W,
                    "h": H,
                    "boxes": kept_boxes_list,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                count += 1
                if count % 100 == 0 or count == len(to_process):
                    elapsed = time_mod.perf_counter() - t0
                    fps = count / elapsed if elapsed > 0 else 0
                    print(f"Pre-labeled {count}/{len(to_process)} images ({fps:.1f} fps)...")

    print(f"Pre-labeling complete. Detections saved to: {detections_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: export
# --------------------------------------------------------------------------- #

def cmd_export(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    threshold = float(args.threshold)

    config_path = root / "candidates" / "config.json"
    manifest_path = root / "candidates" / "manifest.csv"
    detections_path = root / "candidates" / "detections.jsonl"

    if not config_path.exists():
        sys.exit(f"Error: config.json not found at {config_path}")
    if not manifest_path.exists():
        sys.exit(f"Error: manifest.csv not found at {manifest_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    quotas: dict[str, int] = cfg["quotas"]

    # Read manifest rows where status == ok
    manifest_rows: dict[str, dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] == "ok":
                manifest_rows[r["id"]] = r

    # Read detections and filter boxes >= threshold
    detections_by_id: dict[str, dict[str, Any]] = {}
    if detections_path.exists():
        with open(detections_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        detections_by_id[d["id"]] = d
                    except json.JSONDecodeError:
                        pass

    # Build frame info with counts
    frames_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r_id, r in manifest_rows.items():
        det = detections_by_id.get(r_id, {"w": 1920, "h": 1080, "boxes": []})
        w = det.get("w", 1920)
        h = det.get("h", 1080)
        boxes = det.get("boxes", [])

        # Filter by threshold
        kept_boxes = [b for b in boxes if len(b) >= 6 and b[5] >= threshold]

        n_car = sum(1 for b in kept_boxes if b[4] == 0)
        n_motorcycle = sum(1 for b in kept_boxes if b[4] == 1)
        n_truck = sum(1 for b in kept_boxes if b[4] == 2)
        n_bus = sum(1 for b in kept_boxes if b[4] == 3)
        n_total = len(kept_boxes)

        f_info = {
            "id": r_id,
            "cam_id": r["cam_id"],
            "period": r["period"],
            "timestamp": r["timestamp"],
            "session_date": r["session_date"],
            "video": r["video"],
            "offset_s": r["offset_s"],
            "image": r["image"],
            "w": w,
            "h": h,
            "boxes": kept_boxes,
            "n_car": n_car,
            "n_motorcycle": n_motorcycle,
            "n_truck": n_truck,
            "n_bus": n_bus,
            "n_total": n_total,
        }
        key = (r["cam_id"], r["period"])
        frames_by_group.setdefault(key, []).append(f_info)

    # Selection per (cam_id, period)
    selected_frames: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []

    for cam_id in CAMERAS:
        q = quotas[cam_id]
        for period in ("day", "night"):
            group_frames = frames_by_group.get((cam_id, period), [])
            nonempty = [f for f in group_frames if f["n_total"] > 0]
            empty = [f for f in group_frames if f["n_total"] == 0]

            rare = [f for f in nonempty if (f["n_bus"] + f["n_truck"]) > 0]
            # sort by (n_bus > 0, n_bus + n_truck) descending, then timestamp ascending
            rare.sort(key=lambda x: (x["n_bus"] > 0, x["n_bus"] + x["n_truck"], -datetime.fromisoformat(x["timestamp"]).timestamp()), reverse=True)

            max_rare = min(len(rare), math.ceil(q / 2.0))
            chosen = rare[:max_rare]
            chosen_ids = {f["id"] for f in chosen}

            rem_nonempty = [f for f in nonempty if f["id"] not in chosen_ids]
            fill_rng = random.Random(f"42-{cam_id}-{period}-fill")
            fill_rng.shuffle(rem_nonempty)

            needed = q - len(chosen)
            take_from_nonempty = rem_nonempty[:needed]
            chosen.extend(take_from_nonempty)

            needed = q - len(chosen)
            if needed > 0:
                fill_rng.shuffle(empty)
                take_from_empty = empty[:needed]
                chosen.extend(take_from_empty)

            shortfall = q - len(chosen)
            shortfalls.append({
                "cam_id": cam_id,
                "period": period,
                "quota": q,
                "selected": len(chosen),
                "shortfall": max(0, shortfall),
            })
            selected_frames.extend(chosen)

    # Split by session_date per camera
    frames_by_cam: dict[str, list[dict[str, Any]]] = {}
    for f in selected_frames:
        frames_by_cam.setdefault(f["cam_id"], []).append(f)

    for cam_id, c_frames in frames_by_cam.items():
        distinct_dates = sorted(list({f["session_date"] for f in c_frames}))
        val_dates = {d for i, d in enumerate(distinct_dates) if i % 5 == 4}
        for f in c_frames:
            f["split"] = "val" if f["session_date"] in val_dates else "train"

    # Setup ultralytics output directory
    ultralytics_dir = root / "ultralytics"
    _safe_rmtree(ultralytics_dir, root)
    for sp in ("train", "val"):
        (ultralytics_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ultralytics_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # Write images and labels
    print(f"\nWriting Ultralytics dataset to {ultralytics_dir}...")
    for f in selected_frames:
        sp = f["split"]
        src_img = root / f["image"]
        dst_img = ultralytics_dir / "images" / sp / f"{f['id']}.jpg"
        dst_lbl = ultralytics_dir / "labels" / sp / f"{f['id']}.txt"

        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label lines
        lines: list[str] = []
        W = f["w"]
        H = f["h"]
        for b in f["boxes"]:
            cls_id = int(b[4])
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            cx = (x1 + x2) / (2.0 * W)
            cy = (y1 + y2) / (2.0 * H)
            bw = (x2 - x1) / float(W)
            bh = (y2 - y1) / float(H)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(dst_lbl, "w", encoding="utf-8") as lf:
            if lines:
                lf.write("\n".join(lines) + "\n")

    # Write data.yaml
    yaml_path = ultralytics_dir / "data.yaml"
    posix_path = ultralytics_dir.resolve().as_posix()
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {posix_path}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("names:\n")
        for idx, cname in enumerate(CLASSES):
            yf.write(f"  {idx}: {cname}\n")

    # Write selection.csv
    sel_csv_path = ultralytics_dir / "selection.csv"
    with open(sel_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cam_id", "period", "split", "timestamp", "session_date", "video", "offset_s", "n_car", "n_motorcycle", "n_truck", "n_bus"])
        for f in selected_frames:
            writer.writerow([
                f["id"],
                f["cam_id"],
                f["period"],
                f["split"],
                f["timestamp"],
                f["session_date"],
                f["video"],
                f["offset_s"],
                f["n_car"],
                f["n_motorcycle"],
                f["n_truck"],
                f["n_bus"],
            ])

    # Summary tables
    print("\n=== SELECTION PER CAMERA & PERIOD ===")
    print(f"{'Camera':<20} {'Period':<6} {'Quota':<6} {'Selected':<8} {'Shortfall':<8}")
    print("-" * 52)
    for s in shortfalls:
        print(f"{s['cam_id']:<20} {s['period']:<6} {s['quota']:<6} {s['selected']:<8} {s['shortfall']:<8}")

    print("\n=== TOTALS PER PERIOD & SPLIT ===")
    period_split_counts: dict[tuple[str, str], int] = {}
    for f in selected_frames:
        k = (f["period"], f["split"])
        period_split_counts[k] = period_split_counts.get(k, 0) + 1

    print(f"{'Period':<8} {'Split':<8} {'Frames':<8}")
    print("-" * 26)
    for p in ("day", "night"):
        for sp in ("train", "val"):
            cnt = period_split_counts.get((p, sp), 0)
            print(f"{p:<8} {sp:<8} {cnt:<8}")

    print("\n=== BOX COUNTS PER CLASS & PERIOD ===")
    box_counts: dict[tuple[str, str], int] = {}
    for f in selected_frames:
        p = f["period"]
        for b in f["boxes"]:
            cid = int(b[4])
            cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class_{cid}"
            box_counts[(p, cname)] = box_counts.get((p, cname), 0) + 1

    print(f"{'Period':<8} {'Class':<12} {'Boxes':<8}")
    print("-" * 30)
    for p in ("day", "night"):
        for cname in CLASSES:
            cnt = box_counts.get((p, cname), 0)
            print(f"{p:<8} {cname:<12} {cnt:<8}")

    print(f"\nExport complete. data.yaml: {yaml_path}")
    return 0


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m script_tools.build_vehicle_dataset",
        description="Sample CCTV recordings, zero-shot prelabel with Grounding DINO, and export to Ultralytics YOLO format."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_ROOT})",
    )

    # Subcommand: sample
    p_sample = sub.add_parser("sample", parents=[common], help="deterministic candidate pool and frame extraction")
    p_sample.add_argument("--total", type=int, default=4000, help="total target frames across all cameras (default: 4000)")
    p_sample.add_argument("--pool-factor", type=int, default=3, help="candidate pool oversampling factor (default: 3)")
    p_sample.add_argument("--workers", type=int, default=4, help="ffmpeg thread pool workers (default: 4)")
    p_sample.add_argument("--dry-run", action="store_true", help="compute quotas, intervals, and candidates without extracting")
    p_sample.set_defaults(fn=cmd_sample)

    # Subcommand: prelabel
    p_prelabel = sub.add_parser("prelabel", parents=[common], help="Grounding DINO zero-shot detections on CUDA")
    p_prelabel.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-base", help="Hugging Face model ID")
    p_prelabel.add_argument("--floor", type=float, default=0.20, help="detection confidence floor to save (default: 0.20)")
    p_prelabel.set_defaults(fn=cmd_prelabel)

    # Subcommand: export
    p_export = sub.add_parser("export", parents=[common], help="class-aware selection, split, and Ultralytics layout")
    p_export.add_argument("--threshold", type=float, default=0.30, help="export detection threshold (default: 0.30)")
    p_export.set_defaults(fn=cmd_export)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
