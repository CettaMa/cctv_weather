"""Frame resolusi penuh pada menit-menit tertentu sebuah video, untuk memeriksa jalan basah/hujan.

    .venv\\Scripts\\python.exe -m forecast.frame_cek <video_stem> <menit_ke> [<menit_ke> ...]

menit_ke = menit sejak awal video (boleh desimal). Hasil: forecast/data/lembar/<video>_cek.jpg
(grid 3 kolom, 480x360 per frame, label jam WIB menurut nama file).
"""
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

F = Path(__file__).resolve().parent
ROOT = F.parent
sys.path.insert(0, str(ROOT))
from pkji.common import parse_video_metadata  # noqa: E402


def frame(video: Path, detik: float) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "quiet", "-ss", f"{detik:.1f}", "-i", str(video), "-frames:v", "1",
           "-vf", "scale=480:360", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw[:480 * 360 * 3], np.uint8).reshape(360, 480, 3).copy() if len(raw) >= 480 * 360 * 3 \
        else np.zeros((360, 480, 3), np.uint8)


def main() -> None:
    stem, menit = sys.argv[1], [float(x) for x in sys.argv[2:]]
    video = next((ROOT / "dataset").rglob(stem + ".mp4"))
    start = parse_video_metadata(video)[1]
    tiles = []
    for m in menit:
        im = frame(video, m * 60)
        cv2.rectangle(im, (0, 330), (80, 360), (0, 0, 0), -1)
        cv2.putText(im, f"{start + timedelta(minutes=m):%H.%M}", (4, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    out = F / "data" / "lembar" / f"{stem}_cek.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
