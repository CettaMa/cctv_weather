"""Deteksi rekaman berulang atau beku lewat jam pada overlay CCTV.

Jalankan dari folder `extractions` (butuh ffmpeg di PATH):
    .venv\\Scripts\\python.exe eda\\pindai_ulang.py

Latar: sebagian video berisi cuplikan yang sama diputar berulang (misalnya 25 Nov, jam overlay
bertahan di 10:26-10:28 selama sekitar satu jam). Arus dari menit seperti itu bukan pengamatan
baru. Satu keyframe tiap ~10 detik diambil, jam "HH:MM:SS" dibaca dengan pencocokan templat, lalu
sebuah sampel dianggap ulangan bila jamnya jatuh di rentang jam yang sudah pernah terekam.
Jam yang benar-benar beku (diam) tidak membentuk runtun sehingga berstatus tidak dapat dinilai.
Keluaran: eda/data/jam_overlay.csv (per sampel) dan eda/data/ulang_per_menit.csv.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parent
sys.path.insert(0, str(ROOT))
from pkji.common import parse_video_metadata  # noqa: E402

CACHE = E / "data" / "jam"
STEP_KEY = 5                      # keyframe ada tiap 2 detik, jadi satu sampel per 10 detik
X0, Y0, W, H = 551, 11, 68, 16    # kotak jam pada frame 640x480
POS = [0, 1, 3, 4, 6, 7]          # posisi digit pada "HH:MM:SS" (lebar huruf 8 px)
MIN_SKOR = 0.55                   # korelasi minimum tiap digit
# video acuan -> jam yang terbaca pada sampel fps=1/120 (diperiksa mata) untuk membuat templat
ACUAN = {
    "TERUSAN_BUAHBATU_20261125T034400Z": [
        "102706", "102656", "102645", "102808", "102758", "102747", "102814", "102804", "102753", "102743",
        "102732", "102722", "102711", "102700", "102650", "102813", "102803", "102752", "102742", "102808", "102758"],
    "TERUSAN_BUAHBATU_20261125T030117Z": ["094858", "095058", "095258", "095512", "095710", "095912", "100112"],
}


def baca_frame(video: Path, vf: str, keyframe: bool) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "quiet", "-threads", "2"] + (["-skip_frame", "nokey"] if keyframe else []) + [
        "-i", str(video), "-map", "0:v:0", "-vf", f"{vf},scale=640:480,crop={W}:{H}:{X0}:{Y0}",
        "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[:n * W * H], np.uint8).reshape(n, H, W)


def sel(img: np.ndarray, pos: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Ciri satu digit: besar gradien (tidak peduli teks hitam atau putih), dinormalkan."""
    x, y = 2 + pos * 8 + dx, 1 + dy
    c = img[y:y + 14, x:x + 8].astype(np.float32)
    g = np.abs(cv2.Sobel(c, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(c, cv2.CV_32F, 0, 1))
    g -= g.mean()
    return (g / (np.linalg.norm(g) + 1e-6)).ravel()


def templat() -> tuple[np.ndarray, np.ndarray]:
    vec, lab = [], []
    for stem, jam in ACUAN.items():
        video = next((ROOT / "dataset").rglob(stem + ".mp4"))
        frames = baca_frame(video, "fps=1/120", False)
        for img, teks in zip(frames, jam):
            for p, ch in zip(POS, teks):
                vec.append(sel(img, p))
                lab.append(int(ch))
    return np.stack(vec), np.array(lab)


def baca_jam(img: np.ndarray, T: np.ndarray, L: np.ndarray) -> tuple[int, float]:
    """(detik sejak tengah malam atau -1, korelasi digit terlemah)."""
    best = (-1, 0.0)
    for dx in (0, -1, 1):
        for dy in (0, -1, 1):
            digit, lemah = [], 1.0
            for p in POS:
                s = T @ sel(img, p, dx, dy)
                k = int(s.argmax())
                digit.append(int(L[k]))
                lemah = min(lemah, float(s[k]))
            h, m, s_ = (digit[0] * 10 + digit[1], digit[2] * 10 + digit[3], digit[4] * 10 + digit[5])
            if lemah > best[1] and h < 24 and m < 60 and s_ < 60:
                best = (h * 3600 + m * 60 + s_, lemah)
        if best[1] > 0.9:
            break
    return best if best[1] >= MIN_SKOR else (-1, best[1])


def pindai(video: Path, T: np.ndarray, L: np.ndarray) -> np.ndarray:
    out = CACHE / (video.stem + ".npy")
    if out.exists():
        return np.load(out)
    frames = baca_frame(video, f"select='not(mod(n\\,{STEP_KEY}))'", True)
    hasil = np.array([baca_jam(f, T, L) for f in frames], np.float64).reshape(-1, 2)
    np.save(out, hasil)
    return hasil


def runtun(t: np.ndarray, jam: np.ndarray, minimal: int = 3) -> list[np.ndarray]:
    """Kelompok sampel yang jamnya maju sejalan dengan waktu berkas (selisih berubah <= 12 detik).

    Salah baca satu atau dua sampel tidak membentuk runtun; sampel di antaranya dilewati.
    """
    grup: list[list[int]] = []
    for i in np.flatnonzero(~np.isnan(jam)):
        for g in reversed(grup[-6:]):
            k = g[-1]
            if 0 < t[i] - t[k] <= 60 and abs((jam[i] - t[i]) - (jam[k] - t[k])) <= 12:
                g.append(i)
                break
        else:
            grup.append([i])
    return sorted((np.array(g) for g in grup if len(g) >= minimal), key=lambda g: g[0])


def tandai_ulang(t: np.ndarray, jam: np.ndarray, tol: float = 3.0, jeda: float = 45.0) -> np.ndarray:
    """t = waktu urut menurut berkas (detik), jam = waktu overlay absolut (detik, NaN bila tak terbaca).

    Hanya sampel di dalam runtun yang dinilai. Sampel berstatus ulangan bila jam overlay-nya sudah
    dicakup runtun yang terekam >= `jeda` detik sebelumnya (waktu berkas).
    Hasil: 1 ulangan, 0 baru, -1 tidak dapat dinilai.
    """
    status = np.full(len(t), -1, np.int8)
    rentang: list[tuple[float, float, float]] = []      # (jam awal, jam akhir, waktu berkas dibuat)
    pasangan = []                                       # (indeks, indeks sebelumnya dalam runtun)
    for g in runtun(t, jam):
        pasangan += [(i, None) for i in g[:1]] + list(zip(g[1:], g[:-1]))
    for i, k in sorted(pasangan, key=lambda p: p[0]):
        lama = any(a - tol <= jam[i] <= b + tol and t[i] - dibuat >= jeda for a, b, dibuat in rentang[-4000:])
        status[i] = int(lama)
        if k is not None and jam[i] > jam[k]:
            rentang.append((jam[k], jam[i], t[i]))
    return status


def main() -> None:
    import csv
    CACHE.mkdir(parents=True, exist_ok=True)
    T, L = templat()
    videos = sorted((ROOT / "dataset").rglob("*.mp4"), key=lambda p: parse_video_metadata(p)[1])
    with ThreadPoolExecutor(6) as pool:
        semua = list(pool.map(lambda v: pindai(v, T, L), videos))

    epoch = parse_video_metadata(videos[0])[1].replace(hour=0, minute=0, second=0, microsecond=0)
    t, jam, waktu, nama = [], [], [], []
    for video, hasil in zip(videos, semua):
        mulai = parse_video_metadata(video)[1]
        for k, (detik, _) in enumerate(hasil):
            w = mulai + timedelta(seconds=k * STEP_KEY * 2)
            f = (w - epoch).total_seconds()
            if detik < 0:
                a = np.nan
            else:                      # tanggal overlay: yang paling dekat dengan waktu berkas - 13 menit
                acu = f - 13 * 60
                a = detik + 86400 * round((acu - detik) / 86400)
                # rekaman hanya bisa tertinggal dari jam berkas; jam "di masa depan" pasti salah baca
                if not -6 * 3600 <= a - f <= 60:
                    a = np.nan
            t.append(f); jam.append(a); waktu.append(w); nama.append(video.stem)
    t, jam = np.array(t), np.array(jam)
    urut = np.argsort(t, kind="stable")
    status = np.empty(len(t), np.int8)
    status[urut] = tandai_ulang(t[urut], jam[urut])

    with open(E / "data" / "jam_overlay.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "waktu_berkas", "jam_overlay", "selisih_detik", "ulang"])
        for i in urut:
            a = jam[i]
            w.writerow([nama[i], waktu[i].isoformat(),
                        "" if np.isnan(a) else (epoch + timedelta(seconds=a)).isoformat(),
                        "" if np.isnan(a) else round(a - t[i]), status[i]])
    menit: dict[str, list[int]] = {}
    for i in urut:
        m = menit.setdefault(waktu[i].replace(second=0, microsecond=0).isoformat(), [0, 0, 0])
        m[0] += 1
        m[1] += status[i] >= 0
        m[2] += status[i] == 1
    with open(E / "data" / "ulang_per_menit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["menit", "sampel", "terbaca", "ulang", "porsi_ulang"])
        for k, (n, b, u) in sorted(menit.items()):
            w.writerow([k, n, b, u, round(u / b, 4) if b else 0.0])
    terbaca = (status >= 0).mean()
    print(f"video {len(videos)} | sampel {len(t)} | jam terbaca {terbaca:.1%} | sampel ulangan {(status == 1).mean():.1%}")


if __name__ == "__main__":
    main()
