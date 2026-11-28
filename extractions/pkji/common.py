"""Helper nama file, waktu, kelas kendaraan, dan CSV."""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import CLASS_KEYS, IGNORED_ERROR_MESSAGES, TIMEZONE, VIDEO_EXTENSIONS


def summarize_error_messages(messages: Iterable[str]) -> str:
    """Merangkum pesan error unik dengan format 'pesan (Nx)' dan menyaring pesan yang diabaikan."""
    err_counter: Counter = Counter()
    for raw in messages:
        if not raw:
            continue
        for part in str(raw).split("|"):
            cleaned = part.strip()
            if not cleaned or cleaned in IGNORED_ERROR_MESSAGES:
                continue
            # Jika pesan sudah berbentuk "msg (Nx)", normalisasi kembali sebelum dihitung ulang
            match = re.match(r"^(.*)\s+\((\d+)x\)$", cleaned)
            if match:
                base_msg = match.group(1).strip()
                count = int(match.group(2))
                if base_msg and base_msg not in IGNORED_ERROR_MESSAGES:
                    err_counter[base_msg] += count
            else:
                err_counter[cleaned] += 1

    return " || ".join(
        f"{msg} ({cnt}x)" if cnt > 1 else msg for msg, cnt in err_counter.items()
    )


def map_vehicle_class(class_name: str) -> Optional[str]:
    name = class_name.strip().lower().replace("_", " ").replace("-", " ")
    if "motor" in name or "motorcycle" in name or "motorbike" in name:
        return "motor"
    if "mobil" in name or name in {"car", "auto", "automobile"}:
        return "mobil"
    if "bus" in name or "bis" in name:
        return "bus"
    if "truk" in name or "truck" in name or "lorry" in name:
        return "truk"
    return None


def new_counts() -> Dict[str, int]:
    return {key: 0 for key in CLASS_KEYS}


def parse_video_metadata(video_path: Path) -> Tuple[str, datetime]:
    """Ekstrak nama lokasi & timestamp WIB dari nama file video.

    Contoh nama file:
        TERUSAN_BUAHBATU_20261125T030117Z.mp4
        -> location = 'TERUSAN_BUAHBATU'
        -> start_dt = 2026-11-25 10:01:17+07:00 (WIB)
    """
    stem = video_path.stem

    match_utc = re.search(r"(\d{8}T\d{6})Z", stem, flags=re.IGNORECASE)
    if match_utc:
        dt_utc = datetime.strptime(match_utc.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
        start_dt = dt_utc.astimezone(TIMEZONE)
        location = stem[: match_utc.start()].rstrip("_- ") or stem
        return location, start_dt

    match_local = re.search(r"(\d{8}T\d{6})", stem, flags=re.IGNORECASE)
    if match_local:
        start_dt = datetime.strptime(match_local.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=TIMEZONE
        )
        location = stem[: match_local.start()].rstrip("_- ") or stem
        return location, start_dt

    mtime_dt = datetime.fromtimestamp(video_path.stat().st_mtime, tz=TIMEZONE)
    print(
        f"[WARN] Timestamp tidak ditemukan di nama file '{video_path.name}'. "
        f"Menggunakan mtime file: {mtime_dt.isoformat(timespec='seconds')}",
        flush=True,
    )
    return stem, mtime_dt


def collect_video_files(input_path: Path, recursive: bool = False) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        videos = [
            p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        videos.sort(key=lambda p: (parse_video_metadata(p)[1], p.name))
        return videos

    raise FileNotFoundError(f"Path input video tidak ditemukan: {input_path}")


def open_csv_writer(path: Path, fieldnames: List[str], append: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = (not append) or (not path.exists()) or (path.stat().st_size == 0)
    file_handle = path.open(mode, newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        file_handle.flush()
    return file_handle, writer


def append_csv_row(file_handle, writer: csv.DictWriter, row: dict) -> None:
    writer.writerow(row)
    file_handle.flush()


def timestamp_local(value: Optional[datetime]) -> str:
    return value.isoformat(timespec="seconds") if value else ""
