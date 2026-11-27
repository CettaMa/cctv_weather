"""Deteksi jalan basah / hujan dari video CCTV, per menit.

Jalankan dari folder `extractions` (venv ekstraksi cukup: cv2, numpy; butuh ffmpeg di PATH):
    .venv\\Scripts\\python.exe -m forecast.hujan_video

Satu frame tiap 6 detik (320x240). Per menit kalender, median frame menghapus kendaraan sehingga
tersisa latar jalan. Dari latar itu dihitung fitur kecerahan dan tekstur jalan (ROI kedua arah)
dibanding area lain, serta ketajaman frame mentah (tetes air di lensa membuat gambar kabur).
Latar tiap menit disimpan kecil (160x120 JPEG) untuk pelabelan visual.

Waktu memakai jam yang sama dengan data lalu lintas (nama file + posisi frame), jadi selisih jam
overlay tidak berpengaruh pada jeda hujan -> arus.
Keluaran: forecast/data/hujan_video/<video>.csv dan <video>/<menit>.jpg
"""
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

F = Path(__file__).resolve().parent
ROOT = F.parent
sys.path.insert(0, str(ROOT))
from pkji.common import parse_video_metadata  # noqa: E402

OUT = F / "data" / "hujan_video"
W, H, STEP = 320, 240, 6
KOLOM = ["menit", "n_frame", "terang_jalan", "terang_lain", "rasio_terang", "std_jalan", "tekstur_jalan",
         "tekstur_lain", "rasio_tekstur", "sat_jalan", "kilap_jalan", "tajam_frame"]


def roi_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    for f in ("arah_menjauhi", "arah_mendekati"):
        pts = np.array(json.loads((ROOT / "config" / "usulan" / f"{f}.json").read_text())["points"], np.float32)
        cv2.fillPoly(mask, [np.round(pts * W / 640).astype(np.int32)], 1)
    return mask > 0


ROAD = roi_mask()
LAIN = ~ROAD
LAIN[:24] = False          # teks jam dan lokasi
LAIN[-30:, :160] = False   # teks CCTV di kiri bawah


def fitur(bg: np.ndarray, tajam: float, n: int, menit: str) -> list:
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    tj, tl = g[ROAD].mean(), g[LAIN].mean()
    xj, xl = lap[ROAD].mean(), lap[LAIN].mean()
    # kilap: piksel jalan jauh lebih terang dari median jalan (pantulan lampu pada jalan basah)
    kilap = float((g[ROAD] > np.median(g[ROAD]) + 40).mean())
    return [menit, n, tj, tl, tj / max(tl, 1), g[ROAD].std(), xj, xl, xj / max(xl, 1e-3), s[ROAD].mean(), kilap, tajam]


def proses(video: Path) -> None:
    out_csv = OUT / f"{video.stem}.csv"
    if out_csv.exists():
        return
    start = parse_video_metadata(video)[1]
    cmd = ["ffmpeg", "-v", "quiet", "-threads", "2", "-i", str(video), "-map", "0:v:0",
           "-vf", f"fps=1/{STEP},scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    per_menit: dict[str, list] = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        k = 0
        while len(raw := p.stdout.read(W * H * 3)) == W * H * 3:
            m = (start + timedelta(seconds=k * STEP)).replace(second=0, microsecond=0).isoformat()
            per_menit.setdefault(m, []).append(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
            k += 1
    jpg = OUT / video.stem
    jpg.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, fr in per_menit.items():
        if len(fr) < 5:
            continue
        st = np.stack(fr)
        bg = np.median(st, axis=0).astype(np.uint8)
        tajam = float(np.mean([np.abs(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_32F)).mean() for f in fr]))
        rows.append(fitur(bg, tajam, len(fr), m))
        cv2.imwrite(str(jpg / f"{m[11:16].replace(':', '')}.jpg"), cv2.resize(bg, (160, 120)),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    tmp = out_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(KOLOM)
        w.writerows([[r[0], r[1], *[round(float(x), 4) for x in r[2:]]] for r in rows])
    tmp.replace(out_csv)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    videos = sorted((ROOT / "dataset").rglob("*.mp4"))
    with ThreadPoolExecutor(6) as pool:
        for i, _ in enumerate(pool.map(proses, videos), 1):
            if i % 20 == 0:
                print(f"{i}/{len(videos)}", flush=True)
    print("selesai", len(list(OUT.glob("*.csv"))), "video")


if __name__ == "__main__":
    main()
