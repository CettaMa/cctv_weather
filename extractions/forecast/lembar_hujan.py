"""Lembar kontak latar per menit untuk pelabelan visual jalan basah.

    .venv\\Scripts\\python.exe -m forecast.lembar_hujan <video_stem> [--tiap 3] [--mulai HHMM] [--akhir HHMM]

Menyusun thumbnail latar (forecast/data/hujan_video/<video>/<HHMM>.jpg) dalam grid, dengan label
jam (jam nama file, WIB). Hasil: forecast/data/lembar/<video>.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

F = Path(__file__).resolve().parent
SRC = F / "data" / "hujan_video"
DST = F / "data" / "lembar"


def lembar(stem: str, tiap: int = 3, mulai: str = "", akhir: str = "", kolom: int = 6) -> Path:
    files = sorted((SRC / stem).glob("*.jpg"))
    files = [f for f in files if (not mulai or f.stem >= mulai) and (not akhir or f.stem <= akhir)][::tiap]
    tiles = []
    for f in files:
        im = cv2.imread(str(f))
        cv2.rectangle(im, (0, 104), (44, 120), (0, 0, 0), -1)
        cv2.putText(im, f"{f.stem[:2]}.{f.stem[2:]}", (2, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % kolom:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + kolom]) for i in range(0, len(tiles), kolom)])
    DST.mkdir(parents=True, exist_ok=True)
    out = DST / f"{stem}{'_' + mulai if mulai else ''}.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--tiap", type=int, default=3)
    ap.add_argument("--mulai", default="")
    ap.add_argument("--akhir", default="")
    a = ap.parse_args()
    print(lembar(a.stem, a.tiap, a.mulai, a.akhir))
