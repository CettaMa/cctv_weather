#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCTV Weather Classification — binary rain/sunny from CCTV video (offline-first).
Zero-training classical wetness scoring keyed on the road ROI: wet-autumn/standing-water
specular reflections dominate rain frames while dry matte asphalt has few bright pixels,
so wetness = mean of 4 normalized cues (spec_ratio, road_gray, sat_mean, lap_var)
relative to per-cue medians calibrated from rain/sunny clips.

Usage:
  python -m script_tools.weather_vision calibrate <rain_clip> <sunny_clip> [--out <json>]
  python -m script_tools.weather_vision classify <clip> [--refs <json>] [--sample-fps <f>]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Fixed, resolution-relative road ROI polygon and text-band exclusion strips
# (SP Buah Batu viewpoint). Polygon corners expressed as (x_frac, y_frac).
ROAD_ROI_REL = [(0.05, 0.35), (0.95, 0.35), (0.98, 0.95), (0.02, 0.95)]
TEXT_MASK_TOP = 0.08      # excludes "DINAS PERHUBUNGAN" + timestamp band
TEXT_MASK_BOTTOM = 0.92   # excludes "SP. BUAH BATU" band

SPEC_REF = 200.0          # BGR gray threshold: pixels above = wet specular highlight
WORK_SIZE = (320, 240)    # feature-resolution resize

DEFAULT_REFS_PATH = BASE_DIR / "weather" / "thresholds_sp_buahbatu.json"
# Populated at first-classify time from DEFAULT_REFS_PATH when present; the
# module carries DEFAULT_REFS for CLI use when the calibrated file is missing.
DEFAULT_REFS = {
    "road_gray": {"rain": 0.0, "sunny": 0.0},
    "sat_mean": {"rain": 0.0, "sunny": 0.0},
    "lap_var": {"rain": 0.0, "sunny": 0.0},
    "spec_ratio": {"rain": 0.0, "sunny": 0.0},
    "threshold": 0.5,
}


def _abort(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def roi_mask(
    h: int,
    w: int,
    roi_rel: tuple | list = ROAD_ROI_REL,
    text_top: float = TEXT_MASK_TOP,
    text_bottom: float = TEXT_MASK_BOTTOM,
) -> np.ndarray:
    """Boolean mask = road polygon minus text-band strips, at (h, w) size."""
    poly = np.array(
        [(int(x * w), int(y * h)) for x, y in roi_rel], dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    top = int(text_top * h)
    bottom = int(text_bottom * h)
    if top > 0:
        mask[:top, :] = 0
    if bottom < h:
        mask[bottom:, :] = 0
    return mask.astype(bool)


def sample_clip_frames_indexed(
    path: str | Path, n: int = 60
) -> tuple[list[np.ndarray], list[int]]:
    """
    Uniformly sample `n` BGR frames from the clip at native resolution.
    Returns (frames, frame_indices) where frame_indices are decoded frame indices.
    Skip-and-continue on decode failure. Unreadable path or zero decodable
    frames -> SystemExit(2).
    """
    vpath = str(path)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        _abort(f"cannot decode {vpath}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        _abort(f"cannot determine frame count of {vpath}")

    if n > total:
        n = total
    idxs = np.linspace(0, total - 1, n, dtype=np.int64).tolist()

    frames: list[np.ndarray] = []
    indices: list[int] = []
    seen = 0
    while seen < total and len(frames) < n:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if seen in idxs and frame is not None:
            frames.append(frame)
            indices.append(seen)
        seen += 1

    cap.release()
    if not frames:
        _abort(f"cannot decode {vpath}")
    return frames, indices


def sample_clip_frames(path: str | Path, n: int = 60) -> list[np.ndarray]:
    """
    Uniformly sample `n` BGR frames from the clip at native resolution.
    Skip-and-continue on decode failure. Unreadable path or zero decodable
    frames -> SystemExit(2).
    """
    return sample_clip_frames_indexed(path, n)[0]


def extract_frame_features(
    frame: np.ndarray,
    roi_rel: tuple | list = ROAD_ROI_REL,
    text_top: float = TEXT_MASK_TOP,
    text_bottom: float = TEXT_MASK_BOTTOM,
) -> dict:
    """Cues from one frame, computed inside the road ROI at WORK_SIZE."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    mask = roi_mask(sh, sw, roi_rel=roi_rel, text_top=text_top, text_bottom=text_bottom)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    roi_gray = gray[mask]
    roi_sat = sat[mask]

    spec = float(np.mean(gray > SPEC_REF))
    road_gray = float(np.mean(roi_gray)) if roi_gray.size else 0.0
    sat_mean = float(np.mean(roi_sat)) if roi_sat.size else 0.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap[mask])) if roi_gray.size else 0.0

    return {
        "spec_ratio": spec,
        "road_gray": road_gray,
        "sat_mean": sat_mean,
        "lap_var": lap_var,
    }


def _norm_cue(v: float, med_sunny: float, med_rain: float) -> float:
    """Normalize v to 0..1; 1 = closer to rain median, 0 = closer to sunny median."""
    if med_rain == med_sunny:
        return None  # non-discriminative, drop from mean
    d = med_sunny - med_rain
    if med_rain > med_sunny:
        # rain cue higher than sunny (spec_ratio, lap_var)
        s = (v - med_sunny) / d
    else:
        # rain cue lower than sunny (road_gray, sat_mean)
        s = (med_sunny - v) / d
    return float(np.clip(s, 0.0, 1.0))


def score_frame(feats: dict, refs: dict) -> tuple[float, list[str]]:
    """Wetness score = mean of discriminative cues; returns (score, used_cues)."""
    scores: list[float] = []
    used: list[str] = []
    for cue in ("spec_ratio", "road_gray", "sat_mean", "lap_var"):
        if cue not in feats or cue not in refs:
            continue
        med_rain = refs[cue]["rain"]
        med_sunny = refs[cue]["sunny"]
        s = _norm_cue(feats[cue], med_sunny, med_rain)
        if s is not None:
            scores.append(s)
            used.append(cue)
    if not scores:
        return 0.0, []
    return float(np.mean(scores)), used


def classify_frame(feats: dict, refs: dict, threshold: float = 0.5) -> tuple[str, float]:
    s, _ = score_frame(feats, refs)
    return ("rain", s) if s >= threshold else ("sunny", s)


def calibrate(rain_clip: str, sunny_clip: str, out: str) -> dict:
    """
    Measure per-cue medians from 60 sampled frames per clip, sweep threshold on
    those 120 frames, write JSON. Division-by-zero guard: exclude non-discriminative cues.
    """
    rain_frames = sample_clip_frames(rain_clip, n=60)
    sunny_frames = sample_clip_frames(sunny_clip, n=60)

    rain_feats = [extract_frame_features(f) for f in rain_frames]
    sunny_feats = [extract_frame_features(f) for f in sunny_frames]
    labels = ["rain"] * len(rain_feats) + ["sunny"] * len(sunny_feats)

    refs = {"threshold": 0.5}
    dropped: list[str] = []
    for cue in ("road_gray", "sat_mean", "lap_var", "spec_ratio"):
        rain_med = float(np.median([f[cue] for f in rain_feats]))
        sunny_med = float(np.median([f[cue] for f in sunny_feats]))
        if rain_med == sunny_med:
            dropped.append(cue)
            print(f"WARN: cue {cue} non-discriminative, excluded")
            continue
        refs[cue] = {"rain": rain_med, "sunny": sunny_med}

    all_feats = rain_feats + sunny_feats

    def accuracy(thr: float) -> float:
        ok = 0
        for f, lab in zip(all_feats, labels):
            pred, _ = classify_frame(f, refs, thr)
            if pred == lab:
                ok += 1
        return ok / len(all_feats)

    best_thr, best_acc = 0.5, accuracy(0.5)
    # ties -> closest to 0.5
    for thr in np.arange(0.05, 1.0, 0.05):
        acc = accuracy(float(thr))
        if acc > best_acc or (acc == best_acc and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_thr, best_acc = float(thr), acc
    refs["threshold"] = best_thr

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refs, indent=2), encoding="utf-8")

    print(f"Calibration accuracy: {best_acc:.3f} ({int(best_acc * len(all_feats))}/{len(all_feats)} frames) @ threshold {best_thr:.2f}")
    for cue in refs:
        if cue == "threshold":
            continue
        print(f"  {cue}: rain={refs[cue]['rain']:.3f} sunny={refs[cue]['sunny']:.3f}")
    print(f"Wrote {out_path}")
    return refs


def _load_refs(refs_path: str | Path) -> dict:
    p = Path(refs_path) if refs_path else DEFAULT_REFS_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    print(f"WARN: refs file {p} missing, using built-in DEFAULT_REFS", file=sys.stderr)
    return dict(DEFAULT_REFS)


def classify_clip(
    path: str | Path,
    refs_path: str | Path = DEFAULT_REFS_PATH,
    sample_fps: float = 1.0,
) -> dict:
    """
    Classify a clip by temporal vote: median of per-frame wetness scores.
    Returns dict with label / wetness / n_frames / frame_scores.
    Sample rate sample_fps controls frames-per-second (1.0 default).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _abort(f"cannot decode {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    refs = _load_refs(refs_path)

    # Desired frames = sample_fps per real-time second, capped at 300.
    want = int(round(total * sample_fps / fps)) if fps > 0 and total else 0
    n = max(1, min(want, 300, total))
    frames = sample_clip_frames(path, n=n)

    scores: list[float] = []
    for f in frames:
        feats = extract_frame_features(f)
        s, _ = score_frame(feats, refs)
        scores.append(s)

    n_frames = len(scores)
    if n_frames == 0:
        return {"label": "unknown", "wetness": 0.0, "n_frames": 0, "frame_scores": []}

    threshold = refs.get("threshold", 0.5)
    wetness = float(np.median(scores))
    label = "unknown" if n_frames < 5 else ("rain" if wetness >= threshold else "sunny")
    return {
        "label": label,
        "wetness": wetness,
        "n_frames": n_frames,
        "frame_scores": scores,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m script_tools.weather_vision")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="measure cue medians + threshold from two clips")
    p_cal.add_argument("rain_clip")
    p_cal.add_argument("sunny_clip")
    p_cal.add_argument("--out", default=str(BASE_DIR / "weather" / "thresholds_sp_buahbatu.json"))

    p_cls = sub.add_parser("classify", help="classify a clip as rain/sunny")
    p_cls.add_argument("clip")
    p_cls.add_argument("--refs")
    p_cls.add_argument("--sample-fps", type=float, default=1.0)

    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        calibrate(args.rain_clip, args.sunny_clip, args.out)
        return 0

    if args.cmd == "classify":
        res = classify_clip(args.clip, args.refs or DEFAULT_REFS_PATH, args.sample_fps)
        out = f"{res['label']} {res['wetness']:.3f} ({res['n_frames']} frames)"
        print(out)
        if res["label"] == "unknown" and res["n_frames"] < 5:
            print("WARN: too few decodable frames to classify", file=sys.stderr)
            return 1
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())