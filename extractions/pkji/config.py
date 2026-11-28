"""Konstanta, parameter PKJI per lokasi, dan konfigurasi ROI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = BASE_DIR / "models" / "v-1-1.onnx"
DEFAULT_ENGINE_PATH = BASE_DIR / "models" / "v-1-1.engine"
DEFAULT_PT_PATH = BASE_DIR / "model-v1-1.pt"
DEFAULT_ROI_CONFIG_PATH = BASE_DIR / "config" / "roi_config.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "results_video"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".flv", ".webm"}
IGNORED_ERROR_MESSAGES = {
    "Pembacaan frame terputus; membuka ulang stream",
}

# Interval agregasi tambahan (menit) selain ringkasan per video & 1 menit
MULTI_INTERVAL_MINUTES: Tuple[int, ...] = (5, 15, 20, 25, 30, 35, 40, 50)
ALL_AGGREGATION_INTERVALS: Tuple[int, ...] = MULTI_INTERVAL_MINUTES + (60,)

try:
    TIMEZONE = ZoneInfo("Asia/Jakarta")
except ZoneInfoNotFoundError:
    TIMEZONE = timezone(timedelta(hours=7))

# Interval inferensi default 0.20 detik (~5 FPS waktu video)
DEFAULT_INFERENCE_INTERVAL_SECONDS = 0.04   # semua frame pada video 25 fps
LEGACY_INFERENCE_INTERVAL_SECONDS = 0.20    # nilai script lama, dipakai --legacy

DEFAULT_IMGSZ = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
DEFAULT_CPU_THREADS = 4

CLASS_KEYS = ["motor", "mobil", "bus", "truk"]
CLASS_LABELS = {"motor": "Motor", "mobil": "Mobil", "bus": "Bus", "truk": "Truk"}


@dataclass(frozen=True)
class PKJIParams:
    """Parameter PKJI 2023 dan ambang analisis. Default = konstanta script lama.

    Bisa ditimpa per lokasi lewat key "pkji" di file ROI.
    """

    emp_low_flow: Dict[str, float] = field(
        default_factory=lambda: {"mobil": 1.00, "bus": 1.30, "truk": 1.30, "motor": 0.40}
    )
    emp_high_flow: Dict[str, float] = field(
        default_factory=lambda: {"mobil": 1.00, "bus": 1.20, "truk": 1.20, "motor": 0.25}
    )
    emp_flow_threshold_per_lane: float = 1050.0
    c0_per_lane: float = 1700.0
    fc_lj: float = 0.96
    active_lanes_in_roi: int = 2
    # Kalibrasi fisik ruang ROI
    roi_length_meters: float = 90.0
    space_per_smp_meters: float = 7.2
    initial_occupancy_threshold_for_high_emp_pct: float = 25.0
    flow_rolling_window_sec: float = 60.0
    min_flow_extrapolation_window_sec: float = 30.0
    min_dwell_for_valid_flow_sec: float = 0.5
    observation_window_sec: float = 4.0
    min_dwell_for_stop_sec: float = 3.0
    stop_disp_threshold_px_sec: float = 4.0
    slow_disp_threshold_px_sec: float = 18.0
    occupancy_padat_pct: float = 25.0
    occupancy_macet_pct: float = 40.0
    stop_ratio_padat: float = 0.20
    stop_ratio_macet: float = 0.38
    dj_padat_threshold: float = 0.45
    dj_macet_threshold: float = 0.85
    traffic_state_smooth_sec: float = 5
    # Hanya dipakai mode default (tracker dan penggabungan antar video)
    track_max_age_sec: float = 0.8
    track_gate: float = 0.6       # jarak maksimum (dalam ukuran kotak) dari posisi prediksi
    track_new_gate: float = 1.0   # idem untuk track yang baru sekali terlihat
    session_max_gap_sec: float = 5.0
    initial_track_grace_sec: float = 1.0

    @property
    def road_capacity_smp_per_hour(self) -> float:
        return self.active_lanes_in_roi * self.c0_per_lane * self.fc_lj

    @property
    def emp_switch_flow_veh_per_hour(self) -> float:
        return self.active_lanes_in_roi * self.emp_flow_threshold_per_lane

    @property
    def roi_max_smp_capacity(self) -> float:
        return (
            self.active_lanes_in_roi
            * (self.roi_length_meters / self.space_per_smp_meters)
            * self.fc_lj
        )

    @classmethod
    def from_dict(cls, overrides: dict) -> "PKJIParams":
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Parameter PKJI tidak dikenal: {sorted(unknown)}")
        return cls(**overrides)


def resolve_roi_path(roi: Path, location: str) -> Path:
    """--roi boleh file, atau folder berisi <LOKASI>.json dengan cadangan roi_config.json."""
    if roi.is_file():
        return roi
    if roi.is_dir():
        for candidate in (roi / f"{location}.json", roi / "roi_config.json"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"File ROI untuk lokasi '{location}' tidak ditemukan: {roi}")


def load_params(config_path: Path) -> PKJIParams:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return PKJIParams.from_dict(data.get("pkji", {}))


def _scale_points(raw, data: dict, frame_width: int, frame_height: int) -> np.ndarray:
    points = np.asarray(raw, dtype=np.float32)
    config_width = data.get("frame_width")
    config_height = data.get("frame_height")

    if config_width and config_height:
        config_width = int(config_width)
        config_height = int(config_height)
        config_ratio = config_width / config_height
        frame_ratio = frame_width / frame_height
        if abs(config_ratio - frame_ratio) / config_ratio > 0.01:
            raise ValueError(
                f"Aspect ratio video ({frame_width}x{frame_height}) berbeda "
                f"dari ROI ({config_width}x{config_height})."
            )
        points[:, 0] *= frame_width / config_width
        points[:, 1] *= frame_height / config_height

    points[:, 0] = np.clip(points[:, 0], 0, frame_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, frame_height - 1)
    return points


def load_roi(
    config_path: Path, frame_width: int, frame_height: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Mengembalikan (poligon ROI, garis hitung atau None) pada resolusi frame."""
    data = json.loads(config_path.read_text(encoding="utf-8"))

    if "points" not in data:
        raise ValueError("roi_config.json harus memiliki key 'points'.")
    raw = np.asarray(data["points"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3:
        raise ValueError("Format ROI tidak valid. Minimal 3 titik [[x1,y1], ...].")
    points = _scale_points(raw, data, frame_width, frame_height)
    polygon = np.round(points).astype(np.int32).reshape((-1, 1, 2))
    if cv2.contourArea(polygon) <= 0:
        raise ValueError("Poligon ROI tidak memiliki luas.")

    line = None
    if data.get("count_line") is not None:
        raw_line = np.asarray(data["count_line"], dtype=np.float32)
        if raw_line.shape != (2, 2):
            raise ValueError("'count_line' harus berisi tepat 2 titik [[x1,y1],[x2,y2]].")
        line = _scale_points(raw_line, data, frame_width, frame_height)
    return polygon, line
