"""Pembacaan video: decode semua frame, ambil sampel sesuai interval."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


class VideoReader:
    """Iterasi sampel (frame_idx, pos_sec, frame BGR).

    Pemilihan sampel memakai frame_idx / fps (sama dengan script lama) supaya
    cache deteksi berlaku untuk mode default maupun --legacy. pos_sec adalah
    timestamp asli frame menurut container.
    """

    def __init__(self, path: Path, interval: float, decode_threads: int = 0,
                 resume_after: int = -1) -> None:
        self.interval = interval
        # frame_idx sampel terakhir yang sudah diproses (lanjut dari checkpoint)
        self.resume_after = resume_after
        params = [cv2.CAP_PROP_N_THREADS, decode_threads] if decode_threads else []
        self.cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
        self.opened = self.cap.isOpened()
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) if self.opened else 0.0
        if fps <= 0.0 or math.isnan(fps):
            if self.opened:
                print("[WARN] FPS video tidak terbaca; menggunakan fallback 25.0 FPS.", flush=True)
            fps = 25.0
        self.fps = fps
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if self.opened else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if self.opened else 0
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if self.opened else 0
        self.frames_read = 0
        self.end_sec = 0.0
        self.decode_errors = 0

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        last_video_sec = None
        last_pos = -1.0
        while self.cap.grab():
            frame_idx = self.frames_read
            self.frames_read += 1
            video_sec = frame_idx / self.fps
            pos = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(pos) or pos <= last_pos:  # timestamp rusak: lanjutkan monoton
                pos = last_pos + 1.0 / self.fps if last_pos >= 0 else video_sec
            last_pos = pos
            self.end_sec = pos + 1.0 / self.fps

            # lewati tanpa retrieve; tetap di-grab berurutan supaya frame identik
            if frame_idx <= self.resume_after:
                if frame_idx == self.resume_after:
                    last_video_sec = video_sec
                continue
            if (
                last_video_sec is not None
                and (video_sec - last_video_sec) < (self.interval - 1e-6)
            ):
                continue
            retrieved, frame = self.cap.retrieve()
            if not retrieved or frame is None:
                self.decode_errors += 1
                continue
            last_video_sec = video_sec
            yield frame_idx, pos, frame

    def close(self) -> None:
        self.cap.release()
