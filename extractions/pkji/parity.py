"""Paritas deteksi dua model pada frame yang sama.

    python -m pkji.parity --video dataset/<tgl>/<file>.mp4 --candidate models/v-1-1.engine
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIDENCE, DEFAULT_ENGINE_PATH, DEFAULT_ONNX_PATH
from .detector import build_detector
from .video import VideoReader


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def count_matches(a: np.ndarray, b: np.ndarray, threshold: float) -> int:
    """Pasangan satu-satu (kelas sama) dengan IoU > threshold."""
    if not len(a) or not len(b):
        return 0
    iou = iou_matrix(a, b) * (a[:, None, 5] == b[None, :, 5])
    matched = 0
    while iou.max() > threshold:
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        iou[i, :] = 0
        iou[:, j] = 0
        matched += 1
    return matched


def settings(model: Path, rect: bool, args) -> dict:
    return {
        "backend": "onnx" if model.suffix == ".onnx" else "tensorrt", "model": str(model),
        "device": "cuda", "gpu_id": 0, "cpu_threads": 4, "imgsz": 640, "rect": rect,
        "conf": args.conf, "iou": 0.5, "batch": 24,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, nargs="+", required=True)
    ap.add_argument("--reference", type=Path, default=DEFAULT_ONNX_PATH)
    ap.add_argument("--candidate", type=Path, default=DEFAULT_ENGINE_PATH)
    ap.add_argument("--rect", action="store_true", help="Input rect 480x640 untuk kandidat")
    ap.add_argument("--frames", type=int, default=600, help="Jumlah frame total")
    ap.add_argument("--step", type=float, default=2.0, help="Jarak antar frame (detik)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--iou-match", type=float, default=0.95)
    args = ap.parse_args()

    frames = []
    per_video = args.frames // len(args.video)
    for video in args.video:
        reader = VideoReader(video, args.step)
        for n, (_, _, frame) in enumerate(reader, 1):
            frames.append(frame)
            if n >= per_video:
                break
        reader.close()

    results = {}
    for name, model, rect in (("reference", args.reference, False),
                              ("candidate", args.candidate, args.rect)):
        detector = build_detector(settings(model, rect, args))
        out = []
        for i in range(0, len(frames), 24):
            out.extend(detector.detect(frames[i : i + 24]))
        results[name] = (out, detector.names)
        del detector

    (ref, names), (cand, _) = results["reference"], results["candidate"]
    total_ref, total_cand = sum(map(len, ref)), sum(map(len, cand))
    print(f"frame={len(frames)} | deteksi referensi={total_ref} kandidat={total_cand}")
    for threshold in (args.iou_match, 0.9, 0.5):
        matched = sum(count_matches(a, b, threshold) for a, b in zip(ref, cand))
        print(f"IoU>{threshold}: {matched / max(1, total_ref):.2%} referensi, "
              f"{matched / max(1, total_cand):.2%} kandidat berpasangan")
    count_ref = Counter(int(c) for d in ref for c in d[:, 5])
    count_cand = Counter(int(c) for d in cand for c in d[:, 5])
    for class_id, name in sorted(names.items()):
        a, b = count_ref[class_id], count_cand[class_id]
        print(f"  {name:<6} referensi={a:<6} kandidat={b:<6} selisih={(b - a) / max(1, a):+.2%}")


if __name__ == "__main__":
    main()
