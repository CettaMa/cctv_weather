"""Mutu sumber per menit: frame rusak (dari video) dan kendaraan parkir di ROI mendekati (dari cache).

Jalankan dari folder `extractions`:
    .venv\\Scripts\\python.exe eda\\pindai_mutu.py        (butuh ffmpeg di PATH)

Hasil pindaian video disimpan di eda/data/npz sehingga tidak diulang.
Keluaran: eda/data/frame_rusak_per_menit.csv dan eda/data/parkir_per_menit.csv.
"""
import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parent
sys.path.insert(0, str(ROOT))
from pkji import cache
from pkji.common import map_vehicle_class, parse_video_metadata

NPZ = E / "data" / "npz"
W, H, STEP = 320, 240, 5          # satu frame per STEP detik
SMEAR_THRESHOLD = 0.05            # diperiksa visual: di bawahnya gambar utuh
# jangkar (tengah bawah) kotak mobil yang parkir di tepi kanan, di dalam ROI mendekati
SPOT_X, SPOT_Y, SPOT_MIN_WIDTH = (555, 600), (292, 330), 45


def smear(g: np.ndarray) -> float:
    """Fraksi blok yang bergaris vertikal atau abu-abu rata (ciri kerusakan H.264 yang berat)."""
    a = g[24:216].astype(np.float32)                      # tanpa teks jam dan lokasi
    b = a.reshape(8, 24, W // 8, 8)
    vert = np.abs(np.diff(b, axis=1)).mean(axis=(1, 3))
    horiz = np.abs(np.diff(b, axis=3)).mean(axis=(1, 3))
    streak = (vert < 0.6) & (horiz > 1.5)
    gray = (np.abs(b.mean(axis=(1, 3)) - 128) < 3) & (b.std(axis=(1, 3)) < 1.0)
    return float((streak | gray).mean())


def scan(video: Path) -> None:
    out = NPZ / (video.stem + ".npz")
    if out.exists():
        return
    cmd = ["ffmpeg", "-v", "quiet", "-threads", "2", "-i", str(video), "-map", "0:v:0",
           "-vf", f"fps=1/{STEP},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    scores = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while len(raw := proc.stdout.read(W * H)) == W * H:
            scores.append(smear(np.frombuffer(raw, np.uint8).reshape(H, W)))
    np.savez_compressed(out, scores=np.array(scores, np.float32))


def minute_of(start, second: float) -> str:
    return (start + timedelta(seconds=float(second))).replace(second=0, microsecond=0).isoformat()


def write(name: str, header: list, rows: list) -> None:
    with open(E / "data" / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    NPZ.mkdir(parents=True, exist_ok=True)
    videos = sorted((ROOT / "dataset").rglob("*.mp4"))
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(scan, videos))

    broken = defaultdict(lambda: [0, 0])
    for video in videos:
        start = parse_video_metadata(video)[1]
        for k, score in enumerate(np.load(NPZ / (video.stem + ".npz"))["scores"]):
            item = broken[minute_of(start, k * STEP)]
            item[0] += 1
            item[1] += score > SMEAR_THRESHOLD
    write("frame_rusak_per_menit.csv", ["menit", "frame_diperiksa", "frame_rusak", "porsi_rusak"],
          [(m, n, bad, round(bad / n, 4)) for m, (n, bad) in sorted(broken.items())])

    roi_file = ROOT / "config" / "usulan" / "arah_mendekati.json"
    roi = np.round(np.array(json.loads(roi_file.read_text(encoding="utf-8"))["points"])).astype(np.int32)
    mask = np.zeros((481, 641), np.uint8)
    cv2.fillPoly(mask, [roi], 1)
    parked = defaultdict(lambda: [0, 0])
    for video in videos:
        files = [p for p in (ROOT / "cache").glob(f"{video.stem}.*.parquet")
                 if not p.name.endswith(".partial.parquet")]
        if not files:
            continue
        det = cache.load(max(files, key=lambda p: p.stat().st_size))   # cache interval terapat
        if not len(det.frame_idx):
            continue
        start = parse_video_metadata(video)[1]
        b = det.boxes
        vehicle = np.isin(b[:, 5].astype(int), [c for c, n in det.names.items() if map_vehicle_class(n)])
        x = np.clip(np.round((b[:, 0] + b[:, 2]) / 2).astype(int), 0, 640)
        y = np.clip(np.round(b[:, 3]).astype(int), 0, 480)
        spot = (vehicle & (mask[y, x] > 0) & (x >= SPOT_X[0]) & (x <= SPOT_X[1])
                & (y >= SPOT_Y[0]) & (y <= SPOT_Y[1]) & (b[:, 2] - b[:, 0] > SPOT_MIN_WIDTH))
        has = np.zeros(len(det.frame_idx), bool)
        has[np.repeat(np.arange(len(det.frame_idx)), np.diff(det.offsets))[spot]] = True
        for second, flag in zip((det.pos_sec // 1).astype(int), has):
            item = parked[(start + timedelta(seconds=int(second))).replace(second=0, microsecond=0).isoformat()]
            item[0] += 1
            item[1] += flag
    write("parkir_per_menit.csv", ["menit", "frame", "frame_parkir", "porsi_parkir"],
          [(m, n, int(p), round(float(p) / n, 4)) for m, (n, p) in sorted(parked.items())])
    print(f"video: {len(videos)} | menit diperiksa: {len(broken)}")


if __name__ == "__main__":
    main()
