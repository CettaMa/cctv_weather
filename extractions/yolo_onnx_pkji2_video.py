#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import re
import signal
import sys
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# 1. PATH & KONFIGURASI DASAR
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "v-1-1.onnx"
DEFAULT_ROI_CONFIG_PATH = BASE_DIR / "config" / "roi_config.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "results_video"

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
DEFAULT_INFERENCE_INTERVAL_SECONDS = 0.20

# Pengaturan inferensi ONNX
DEFAULT_IMGSZ = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
DEFAULT_CPU_THREADS = 4

FALLBACK_CLASS_NAMES = {
    0: "motor",
    1: "mobil",
    2: "bus",
    3: "truk",
}

CLASS_KEYS = ["motor", "mobil", "bus", "truk"]
CLASS_LABELS = {
    "motor": "Motor",
    "mobil": "Mobil",
    "bus": "Bus",
    "truk": "Truk",
}


# ============================================================
# 2. PARAMETER STANDAR PKJI 2023 & METODE EKSTRAKSI 2
# ============================================================
EMP_LOW_FLOW: Dict[str, float] = {
    "mobil": 1.00,
    "bus": 1.30,
    "truk": 1.30,
    "motor": 0.40,
}

EMP_HIGH_FLOW: Dict[str, float] = {
    "mobil": 1.00,
    "bus": 1.20,
    "truk": 1.20,
    "motor": 0.25,
}

PKJI_EMP_FLOW_THRESHOLD_PER_LANE = 1050.0
PKJI_C0_PER_LANE = 1700.0
PKJI_FC_LJ = 0.96
ACTIVE_LANES_IN_ROI = 2

# Kapasitas Efektif Jalan (C = n * C0 * FC_LJ) = 3264.0 SMP/jam
ROAD_CAPACITY_SMP_PER_HOUR = ACTIVE_LANES_IN_ROI * PKJI_C0_PER_LANE * PKJI_FC_LJ
EMP_SWITCH_FLOW_VEH_PER_HOUR = ACTIVE_LANES_IN_ROI * PKJI_EMP_FLOW_THRESHOLD_PER_LANE

# Kalibrasi Fisik Ruang ROI (C_ROI = 24.0 SMP)
ROI_LENGTH_METERS = 90.0
SPACE_PER_SMP_METERS = 7.2
ROI_MAX_SMP_CAPACITY = (
    ACTIVE_LANES_IN_ROI * (ROI_LENGTH_METERS / SPACE_PER_SMP_METERS) * PKJI_FC_LJ
)

INITIAL_OCCUPANCY_THRESHOLD_FOR_HIGH_EMP_PCT = 25.0

FLOW_ROLLING_WINDOW_SEC = 60.0
MIN_FLOW_EXTRAPOLATION_WINDOW_SEC = 30.0
MIN_DWELL_FOR_VALID_FLOW_SEC = 0.5

OBSERVATION_WINDOW_SEC = 4.0
MIN_DWELL_FOR_STOP_SEC = 3.0
STOP_DISP_THRESHOLD_PX_SEC = 4.0
SLOW_DISP_THRESHOLD_PX_SEC = 18.0

OCCUPANCY_PADAT_PCT = 25.0
OCCUPANCY_MACET_PCT = 40.0
STOP_RATIO_PADAT = 0.20
STOP_RATIO_MACET = 0.38

DJ_PADAT_THRESHOLD = 0.45
DJ_MACET_THRESHOLD = 0.85
TRAFFIC_STATE_SMOOTH_SEC = 5


# ============================================================
# 3. SKEMA KOLOM CSV
# ============================================================
VIDEO_SUMMARY_CSV_FIELDS = [
    "video_file",
    "location",
    "video_start_local",
    "video_end_local",
    "video_duration_seconds",
    "video_fps",
    "sampled_frames",
    "dominant_emp_mode",
    "average_active_motor",
    "average_active_mobil",
    "average_active_bus",
    "average_active_truk",
    "average_active_total",
    "max_active_total",
    "avg_smp_frame",
    "c_roi_capacity_smp",
    "avg_roi_occupancy_pct",
    "avg_pixel_occupancy_pct",
    "est_flow_veh_per_hour",
    "est_q_smp_per_hour",
    "road_capacity_c_smp_per_hour",
    "avg_dj_flow",
    "avg_dj_spatial",
    "avg_dj_effective",
    "max_dj_effective",
    "avg_moving_vehicles",
    "avg_slow_vehicles",
    "avg_stopped_vehicles",
    "avg_stopped_ratio",
    "pct_time_lancar",
    "pct_time_padat",
    "pct_time_macet",
    "video_traffic_condition",
    "average_inference_ms",
    "processing_fps",
    "status",
    "error_message",
]

SUBSESSION_CSV_FIELDS = [
    "video_file",
    "location",
    "hour_bucket",
    "one_min_bucket",
    "session_point",
    "sample_start_local",
    "sample_end_local",
    "target_sample_duration_seconds",
    "actual_sample_duration_seconds",
    "sampled_frames",
    "dominant_emp_mode",
    "average_active_motor",
    "average_active_mobil",
    "average_active_bus",
    "average_active_truk",
    "average_active_total",
    "max_active_total",
    "avg_smp_frame",
    "c_roi_capacity_smp",
    "avg_roi_occupancy_pct",
    "avg_pixel_occupancy_pct",
    "est_flow_veh_per_hour",
    "est_q_smp_per_hour",
    "road_capacity_c_smp_per_hour",
    "avg_dj_flow",
    "avg_dj_spatial",
    "avg_dj_effective",
    "max_dj_effective",
    "avg_moving_vehicles",
    "avg_slow_vehicles",
    "avg_stopped_vehicles",
    "avg_stopped_ratio",
    "pct_time_lancar",
    "pct_time_padat",
    "pct_time_macet",
    "dominant_traffic_condition",
    "one_min_traffic_condition",
    "average_inference_ms",
    "processing_fps",
    "status",
    "error_message",
]

INTERVAL_CSV_FIELDS = [
    "interval_minutes",
    "window_start",
    "window_end",
    "expected_sessions",
    "sessions_completed",
    "first_sample_start_local",
    "last_sample_end_local",
    "total_target_duration_seconds",
    "total_actual_duration_seconds",
    "total_sampled_frames",
    "dominant_emp_mode",
    "average_active_motor",
    "average_active_mobil",
    "average_active_bus",
    "average_active_truk",
    "average_active_total",
    "max_active_total",
    "avg_smp_frame",
    "c_roi_capacity_smp",
    "avg_roi_occupancy_pct",
    "avg_pixel_occupancy_pct",
    "est_flow_veh_per_hour",
    "est_q_smp_per_hour",
    "road_capacity_c_smp_per_hour",
    "avg_dj_flow",
    "avg_dj_spatial",
    "avg_dj_effective",
    "max_dj_effective",
    "avg_moving_vehicles",
    "avg_slow_vehicles",
    "avg_stopped_vehicles",
    "avg_stopped_ratio",
    "pct_time_lancar",
    "pct_time_padat",
    "pct_time_macet",
    "window_traffic_condition",
    "average_inference_ms",
    "processing_fps",
    "status",
    "error_message",
]

HOURLY_CSV_FIELDS = [
    "scheduled_hour",
    "sessions_completed",
    "first_sample_start_local",
    "last_sample_end_local",
    "total_target_duration_seconds",
    "total_actual_duration_seconds",
    "total_sampled_frames",
    "dominant_emp_mode",
    "average_active_motor",
    "average_active_mobil",
    "average_active_bus",
    "average_active_truk",
    "average_active_total",
    "max_active_total",
    "avg_smp_frame",
    "c_roi_capacity_smp",
    "avg_roi_occupancy_pct",
    "avg_pixel_occupancy_pct",
    "est_flow_veh_per_hour",
    "est_q_smp_per_hour",
    "road_capacity_c_smp_per_hour",
    "avg_dj_flow",
    "avg_dj_spatial",
    "avg_dj_effective",
    "max_dj_effective",
    "avg_moving_vehicles",
    "avg_slow_vehicles",
    "avg_stopped_vehicles",
    "avg_stopped_ratio",
    "pct_time_lancar",
    "pct_time_padat",
    "pct_time_macet",
    "hourly_traffic_condition",
    "average_inference_ms",
    "processing_fps",
    "status",
    "error_message",
]

STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nSignal {signum} diterima. Menghentikan ekstraksi dengan aman...", flush=True)


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


# ============================================================
# 4. PURE ONNX DETECTOR (CUDA GPU + OPSIONAL FALLBACK CPU)
# ============================================================
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


class PureONNXYOLO:
    """Wrapper ONNX Runtime untuk model YOLO dengan akselerasi CUDA GPU."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        gpu_id: int = 0,
        cpu_threads: int = DEFAULT_CPU_THREADS,
        warmup_imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        # Bantu ONNX Runtime menemukan CUDA/cuDNN DLL di Windows jika diinstal via pip
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                print(f"[WARN] preload_dlls() dilewati: {exc}", flush=True)

        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime version : {ort.__version__}", flush=True)
        print(f"Available providers  : {available_providers}", flush=True)

        cuda_available = "CUDAExecutionProvider" in available_providers
        cuda_provider_entry = (
            "CUDAExecutionProvider",
            {
                "device_id": gpu_id,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        )

        if device == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDAExecutionProvider tidak tersedia di ONNX Runtime!\n"
                    "Pastikan Anda sudah menghapus 'onnxruntime' (CPU) dan menginstal 'onnxruntime-gpu':\n"
                    "  pip uninstall -y onnxruntime\n"
                    "  pip install onnxruntime-gpu\n"
                    "Atau jalankan dengan --device auto / --device cpu jika ingin memakai CPU."
                )
            requested_providers = [cuda_provider_entry, "CPUExecutionProvider"]
        elif device == "auto":
            requested_providers = (
                [cuda_provider_entry, "CPUExecutionProvider"]
                if cuda_available
                else ["CPUExecutionProvider"]
            )
        elif device == "cpu":
            requested_providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Device tidak dikenali: {device}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if device == "cpu" or not cuda_available:
            sess_options.intra_op_num_threads = cpu_threads
            sess_options.inter_op_num_threads = 1

        print(f"Memuat model ONNX    : {model_path}", flush=True)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=requested_providers,
        )

        self.active_providers = self.session.get_providers()
        print(f"Active providers     : {self.active_providers}", flush=True)

        self.using_cuda = "CUDAExecutionProvider" in self.active_providers
        if device == "cuda" and not self.using_cuda:
            raise RuntimeError(
                "ONNX Runtime gagal mengaktifkan CUDAExecutionProvider dan jatuh ke CPU.\n"
                "Penyebab paling umum di Windows: mismatch versi CUDA/cuDNN dengan onnxruntime-gpu.\n"
                "Cek error log di atas, atau jalankan dengan --device auto untuk fallback CPU."
            )

        self.device_label = f"CUDA (GPU #{gpu_id})" if self.using_cuda else "CPU"
        print(f"Device inferensi     : {self.device_label}", flush=True)

        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.input_dtype = (
            np.float16 if "float16" in str(self.input_meta.type).lower() else np.float32
        )
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.names = self._extract_class_names()

        print("Mapping kelas model:", flush=True)
        for class_id, class_name in sorted(self.names.items()):
            vehicle_group = map_vehicle_class(str(class_name))
            target = CLASS_LABELS[vehicle_group] if vehicle_group else "diabaikan"
            print(f"  ID {class_id}: {class_name} -> {target}", flush=True)

        self._warmup(warmup_imgsz)

    def _warmup(self, imgsz: int) -> None:
        """Warmup awal agar kompilasi kernel CUDA tidak mengacaukan rata-rata latency."""
        try:
            dummy = np.zeros((1, 3, imgsz, imgsz), dtype=self.input_dtype)
            t0 = time.perf_counter()
            self.session.run(self.output_names, {self.input_name: dummy})
            t1 = (time.perf_counter() - t0) * 1000.0
            print(f"Warmup {self.device_label} selesai ({t1:.1f} ms)", flush=True)
        except Exception as exc:
            print(f"[WARN] Warmup gagal ({exc}), lanjut inferensi normal.", flush=True)

    def _extract_class_names(self) -> Dict[int, str]:
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                parsed = ast.literal_eval(meta["names"])
                if isinstance(parsed, dict):
                    return {int(k): str(v) for k, v in parsed.items()}
                if isinstance(parsed, list):
                    return {i: str(v) for i, v in enumerate(parsed)}
        except Exception as exc:
            print(f"WARNING: Gagal membaca metadata kelas ONNX ({exc}).", flush=True)
        return dict(FALLBACK_CLASS_NAMES)

    def _letterbox(
        self, frame: np.ndarray, new_size: int = 640
    ) -> Tuple[np.ndarray, float, int, int]:
        orig_h, orig_w = frame.shape[:2]
        scale = min(new_size / orig_h, new_size / orig_w)
        new_unpad_w = int(round(orig_w * scale))
        new_unpad_h = int(round(orig_h * scale))

        dw = (new_size - new_unpad_w) / 2.0
        dh = (new_size - new_unpad_h) / 2.0

        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(
                frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized = frame

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        blob = padded[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=self.input_dtype) / np.asarray(
            255.0, dtype=self.input_dtype
        )
        return blob[None, ...], scale, left, top

    def infer(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> List[Tuple[float, float, float, float, float, int]]:
        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_left, pad_top = self._letterbox(frame, imgsz)

        outputs = self.session.run(self.output_names, {self.input_name: blob})
        pred = np.asarray(outputs[0], dtype=np.float32)

        if pred.ndim == 3 and pred.shape[2] == 6 and pred.shape[1] <= 1000:
            dets = pred[0]
            mask = dets[:, 4] >= conf_thres
            dets = dets[mask][:max_det]

            results: List[Tuple[float, float, float, float, float, int]] = []
            for row in dets:
                x1, y1, x2, y2, score, cls_id = row
                x1 = float(np.clip((x1 - pad_left) / scale, 0, orig_w - 1))
                y1 = float(np.clip((y1 - pad_top) / scale, 0, orig_h - 1))
                x2 = float(np.clip((x2 - pad_left) / scale, 0, orig_w - 1))
                y2 = float(np.clip((y2 - pad_top) / scale, 0, orig_h - 1))
                results.append((x1, y1, x2, y2, float(score), int(cls_id)))
            return results

        preds = pred[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= conf_thres
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        cx = boxes_cxcywh[:, 0]
        cy = boxes_cxcywh[:, 1]
        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        max_wh = 4096.0
        offsets = class_ids.astype(np.float32) * max_wh
        nms_boxes = np.stack([x1 + offsets, y1 + offsets, w, h], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(), conf_thres, iou_thres
        )
        if len(indices) == 0:
            return []

        indices = np.asarray(indices).flatten()[:max_det]
        results = []
        for idx in indices:
            bx1 = float(np.clip((x1[idx] - pad_left) / scale, 0, orig_w - 1))
            by1 = float(np.clip((y1[idx] - pad_top) / scale, 0, orig_h - 1))
            bx2 = float(np.clip((x2[idx] - pad_left) / scale, 0, orig_w - 1))
            by2 = float(np.clip((y2[idx] - pad_top) / scale, 0, orig_h - 1))
            results.append(
                (bx1, by1, bx2, by2, float(confidences[idx]), int(class_ids[idx]))
            )

        return results


# ============================================================
# 5. TRACKER, PKJI 2023 CALCULATOR & MOTION ANALYZER
# ============================================================
class SimpleCentroidTracker:
    """Tracker ringan berbasis jarak titik tengah."""

    def __init__(self, max_disappeared: int = 12, max_distance: float = 135.0) -> None:
        self.next_id = 1
        self.objects: Dict[int, Tuple[float, float]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, centroids: List[Tuple[float, float]]) -> List[int]:
        if not centroids:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return []

        if not self.objects:
            assigned_ids = []
            for c in centroids:
                obj_id = self.next_id
                self.objects[obj_id] = c
                self.disappeared[obj_id] = 0
                assigned_ids.append(obj_id)
                self.next_id += 1
            return assigned_ids

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        dist_matrix = np.zeros((len(obj_centroids), len(centroids)), dtype=np.float32)
        for i, (ox, oy) in enumerate(obj_centroids):
            for j, (cx, cy) in enumerate(centroids):
                dist_matrix[i, j] = math.hypot(ox - cx, oy - cy)

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        col_to_id: Dict[int, int] = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = centroids[col]
            self.disappeared[obj_id] = 0
            col_to_id[col] = obj_id
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(obj_centroids))) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                del self.objects[obj_id]
                del self.disappeared[obj_id]

        assigned_ids = []
        for col in range(len(centroids)):
            if col in col_to_id:
                assigned_ids.append(col_to_id[col])
            else:
                obj_id = self.next_id
                self.objects[obj_id] = centroids[col]
                self.disappeared[obj_id] = 0
                assigned_ids.append(obj_id)
                self.next_id += 1

        return assigned_ids


def new_counts() -> Dict[str, int]:
    return {key: 0 for key in CLASS_KEYS}


class PKJIFlowAndOccupancyCalculator:
    """Menghitung metrik Standar PKJI 2023 (EMP, SMP, Okupansi ROI, Q, dan DJ)."""

    def __init__(self, rolling_window_sec: float = FLOW_ROLLING_WINDOW_SEC) -> None:
        self.rolling_window_sec = rolling_window_sec
        self.counted_track_ids: Set[int] = set()
        self.rolling_departures: deque = deque()

    def register_active_tracks(
        self, timestamp_sec: float, active_tracks: List[Tuple[int, str, float]]
    ) -> None:
        for track_id, v_class, dwell_time in active_tracks:
            if (
                track_id not in self.counted_track_ids
                and dwell_time >= MIN_DWELL_FOR_VALID_FLOW_SEC
            ):
                self.counted_track_ids.add(track_id)
                self.rolling_departures.append((timestamp_sec, track_id, v_class))

        while (
            self.rolling_departures
            and (timestamp_sec - self.rolling_departures[0][0]) > self.rolling_window_sec
        ):
            self.rolling_departures.popleft()

    def compute_pkji_metrics(
        self, timestamp_sec: float, frame_counts: Dict[str, int]
    ) -> Dict[str, float | str]:
        effective_window_sec = min(
            max(MIN_FLOW_EXTRAPOLATION_WINDOW_SEC, timestamp_sec),
            self.rolling_window_sec,
        )
        window_counts = new_counts()
        for _, _, v_class in self.rolling_departures:
            window_counts[v_class] += 1

        scale_to_hour = 3600.0 / effective_window_sec
        hourly_counts = {k: window_counts[k] * scale_to_hour for k in CLASS_KEYS}
        total_flow_veh_per_hour = sum(hourly_counts.values())

        smp_frame_low = sum(frame_counts[k] * EMP_LOW_FLOW[k] for k in CLASS_KEYS)
        occ_low_pct = (smp_frame_low / ROI_MAX_SMP_CAPACITY) * 100.0

        if (
            total_flow_veh_per_hour >= EMP_SWITCH_FLOW_VEH_PER_HOUR
            or occ_low_pct >= INITIAL_OCCUPANCY_THRESHOLD_FOR_HIGH_EMP_PCT
        ):
            active_emp = EMP_HIGH_FLOW
            emp_mode = "HIGH_FLOW (>=1050/lajur)"
        else:
            active_emp = EMP_LOW_FLOW
            emp_mode = "LOW_FLOW (<1050/lajur)"

        smp_frame = sum(frame_counts[k] * active_emp[k] for k in CLASS_KEYS)
        roi_occupancy_pct = min(100.0, (smp_frame / ROI_MAX_SMP_CAPACITY) * 100.0)

        q_flow_smp_per_hour = sum(hourly_counts[k] * active_emp[k] for k in CLASS_KEYS)
        dj_from_flow = q_flow_smp_per_hour / ROAD_CAPACITY_SMP_PER_HOUR

        if roi_occupancy_pct <= OCCUPANCY_PADAT_PCT:
            dj_spatial = (roi_occupancy_pct / OCCUPANCY_PADAT_PCT) * DJ_PADAT_THRESHOLD
        elif roi_occupancy_pct <= OCCUPANCY_MACET_PCT:
            ratio_mid = (roi_occupancy_pct - OCCUPANCY_PADAT_PCT) / (
                OCCUPANCY_MACET_PCT - OCCUPANCY_PADAT_PCT
            )
            dj_spatial = DJ_PADAT_THRESHOLD + ratio_mid * (
                DJ_MACET_THRESHOLD - DJ_PADAT_THRESHOLD
            )
        else:
            ratio_high = (roi_occupancy_pct - OCCUPANCY_MACET_PCT) / (
                100.0 - OCCUPANCY_MACET_PCT
            )
            dj_spatial = DJ_MACET_THRESHOLD + ratio_high * (1.00 - DJ_MACET_THRESHOLD)

        effective_dj = max(dj_from_flow, dj_spatial)

        return {
            "emp_mode": emp_mode,
            "smp_frame": smp_frame,
            "roi_occupancy_pct": roi_occupancy_pct,
            "flow_veh_per_hour": total_flow_veh_per_hour,
            "q_smp_per_hour": q_flow_smp_per_hour,
            "dj_flow": dj_from_flow,
            "dj_spatial": dj_spatial,
            "dj_effective": effective_dj,
        }


class VehicleMotionAnalyzer:
    """Melacak riwayat posisi dan status gerak kendaraan (Moving, SLOW, STOP) di ROI."""

    def __init__(self, observation_window: float = OBSERVATION_WINDOW_SEC) -> None:
        self.observation_window = observation_window
        self.history: Dict[int, deque] = {}
        self.first_seen: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}

    def update_vehicle(
        self, track_id: int, timestamp_sec: float, x: int, y: int
    ) -> Dict[str, float | str]:
        if track_id not in self.history:
            self.history[track_id] = deque()
            self.first_seen[track_id] = timestamp_sec

        self.last_seen[track_id] = timestamp_sec
        hist = self.history[track_id]
        hist.append((timestamp_sec, x, y))

        while len(hist) > 2 and (timestamp_sec - hist[0][0]) > self.observation_window:
            hist.popleft()

        dwell_time = timestamp_sec - self.first_seen[track_id]
        window_dt = timestamp_sec - hist[0][0]

        if len(hist) >= 2 and window_dt > 0.1:
            dx = x - hist[0][1]
            dy = y - hist[0][2]
            displacement_px = math.hypot(dx, dy)
            speed_px_sec = displacement_px / window_dt
        else:
            displacement_px = 0.0
            speed_px_sec = 0.0

        if (
            dwell_time >= MIN_DWELL_FOR_STOP_SEC
            and speed_px_sec < STOP_DISP_THRESHOLD_PX_SEC
        ):
            motion_state = "STOP"
        elif dwell_time >= 0.5 and speed_px_sec < SLOW_DISP_THRESHOLD_PX_SEC:
            motion_state = "SLOW"
        else:
            motion_state = "Moving"

        return {
            "dwell_time": dwell_time,
            "displacement_px": displacement_px,
            "speed_px_sec": speed_px_sec,
            "motion_state": motion_state,
        }

    def cleanup_stale(self, current_time: float, max_idle_sec: float = 5.0) -> None:
        stale_ids = [
            tid
            for tid, last_t in self.last_seen.items()
            if (current_time - last_t) > max_idle_sec
        ]
        for tid in stale_ids:
            self.history.pop(tid, None)
            self.first_seen.pop(tid, None)
            self.last_seen.pop(tid, None)


def classify_traffic_condition(
    occupancy_pct: float,
    dj_effective: float,
    total_vehicles: float,
    moving_count: float,
    slow_count: float,
    stop_count: float,
) -> str:
    """Menentukan kondisi lalu lintas (Lancar, Padat, Macet) sesuai kalibrasi PKJI2."""
    if total_vehicles <= 0:
        return "Lancar (Free Flow)"

    stop_ratio = stop_count / total_vehicles
    slow_or_stop_ratio = (slow_count + stop_count) / total_vehicles

    min_vehicles_padat = 6
    min_vehicles_macet = 10
    min_stopped_padat = 2
    min_stopped_macet = 4

    enough_for_padat = (total_vehicles >= min_vehicles_padat) or (occupancy_pct >= 18.0)
    enough_for_macet = (total_vehicles >= min_vehicles_macet) or (occupancy_pct >= 35.0)

    if (
        (occupancy_pct >= OCCUPANCY_MACET_PCT and slow_or_stop_ratio >= 0.25)
        or (
            enough_for_macet
            and stop_count >= min_stopped_macet
            and stop_ratio >= STOP_RATIO_MACET
        )
        or (
            occupancy_pct >= OCCUPANCY_PADAT_PCT
            and stop_count >= min_stopped_macet
            and stop_ratio >= 0.35
        )
        or (
            enough_for_macet
            and dj_effective >= DJ_MACET_THRESHOLD
            and slow_or_stop_ratio >= 0.40
        )
    ):
        return "Macet (Congestion)"

    if (
        occupancy_pct >= OCCUPANCY_PADAT_PCT
        or (
            enough_for_padat
            and (
                (stop_count >= min_stopped_padat and stop_ratio >= STOP_RATIO_PADAT)
                or slow_or_stop_ratio >= 0.50
                or (dj_effective >= DJ_PADAT_THRESHOLD and occupancy_pct >= 20.0)
            )
        )
    ):
        return "Padat (Slow Traffic)"

    return "Lancar (Free Flow)"


# ============================================================
# 6. AKUMULATOR METRIK (PER 1 MENIT & PER VIDEO)
# ============================================================
class MetricsAccumulator:
    """Mengakumulasi metrik frame demi frame untuk 1 bucket menit atau 1 video penuh."""

    def __init__(self, inference_interval: float) -> None:
        self.inference_interval = inference_interval
        self.sums = new_counts()
        self.motion_sums = {"Moving": 0, "SLOW": 0, "STOP": 0}
        self.status_counter: Counter = Counter()
        self.emp_mode_counter: Counter = Counter()

        self.sampled_frames = 0
        self.max_total = 0
        self.inference_time_total_ms = 0.0
        self.wall_start_mono = time.perf_counter()

        self.smp_sum = 0.0
        self.occ_sum = 0.0
        self.px_occ_sum = 0.0
        self.flow_veh_sum = 0.0
        self.q_smp_sum = 0.0
        self.dj_flow_sum = 0.0
        self.dj_spatial_sum = 0.0
        self.dj_eff_sum = 0.0
        self.max_dj_eff = 0.0

        self.first_video_sec: Optional[float] = None
        self.last_video_sec: Optional[float] = None
        self.first_dt: Optional[datetime] = None
        self.last_dt: Optional[datetime] = None
        self.errors: List[str] = []

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(message)

    def add_frame(
        self,
        video_sec: float,
        frame_dt: datetime,
        frame_counts: Dict[str, int],
        motion_counts: Dict[str, int],
        pkji_metrics: Dict[str, float | str],
        pixel_occupancy_pct: float,
        smoothed_status: str,
        inference_ms: float,
    ) -> None:
        if self.first_video_sec is None:
            self.first_video_sec = video_sec
            self.first_dt = frame_dt
        self.last_video_sec = video_sec
        self.last_dt = frame_dt

        total_in_roi = sum(frame_counts.values())
        self.sampled_frames += 1
        self.inference_time_total_ms += inference_ms
        self.max_total = max(self.max_total, total_in_roi)

        for k in CLASS_KEYS:
            self.sums[k] += frame_counts[k]
        for m_key in ("Moving", "SLOW", "STOP"):
            self.motion_sums[m_key] += motion_counts[m_key]

        dj_eff = float(pkji_metrics["dj_effective"])
        self.smp_sum += float(pkji_metrics["smp_frame"])
        self.occ_sum += float(pkji_metrics["roi_occupancy_pct"])
        self.px_occ_sum += pixel_occupancy_pct
        self.flow_veh_sum += float(pkji_metrics["flow_veh_per_hour"])
        self.q_smp_sum += float(pkji_metrics["q_smp_per_hour"])
        self.dj_flow_sum += float(pkji_metrics["dj_flow"])
        self.dj_spatial_sum += float(pkji_metrics["dj_spatial"])
        self.dj_eff_sum += dj_eff
        self.max_dj_eff = max(self.max_dj_eff, dj_eff)

        self.status_counter[smoothed_status] += 1
        self.emp_mode_counter[str(pkji_metrics["emp_mode"])] += 1

    def video_span_seconds(self) -> float:
        if self.first_video_sec is None or self.last_video_sec is None:
            return 0.0
        return round(
            max(
                self.inference_interval,
                (self.last_video_sec - self.first_video_sec) + self.inference_interval,
            ),
            3,
        )

    def summarize(self) -> Dict[str, object]:
        wall_elapsed = max(1e-6, time.perf_counter() - self.wall_start_mono)
        n = self.sampled_frames

        if n > 0:
            averages = {k: round(self.sums[k] / n, 3) for k in CLASS_KEYS}
            avg_total = round(sum(self.sums.values()) / n, 3)
            avg_moving = round(self.motion_sums["Moving"] / n, 3)
            avg_slow = round(self.motion_sums["SLOW"] / n, 3)
            avg_stop = round(self.motion_sums["STOP"] / n, 3)
            avg_stop_ratio = round(avg_stop / avg_total, 3) if avg_total > 0 else 0.0

            avg_smp = round(self.smp_sum / n, 3)
            avg_occ = round(self.occ_sum / n, 2)
            avg_px_occ = round(self.px_occ_sum / n, 2)
            est_flow_veh = round(self.flow_veh_sum / n, 1)
            est_q_smp = round(self.q_smp_sum / n, 1)
            avg_dj_fl = round(self.dj_flow_sum / n, 3)
            avg_dj_sp = round(self.dj_spatial_sum / n, 3)
            avg_dj_ef = round(self.dj_eff_sum / n, 3)
            max_dj_ef = round(self.max_dj_eff, 3)

            lancar_pct = round(
                self.status_counter["Lancar (Free Flow)"] / n * 100.0, 2
            )
            padat_pct = round(
                self.status_counter["Padat (Slow Traffic)"] / n * 100.0, 2
            )
            macet_pct = round(
                self.status_counter["Macet (Congestion)"] / n * 100.0, 2
            )
            dominant_cond = self.status_counter.most_common(1)[0][0]
            dominant_emp = self.emp_mode_counter.most_common(1)[0][0]
            avg_infer_ms = round(self.inference_time_total_ms / n, 3)
            proc_fps = round(n / wall_elapsed, 3)
        else:
            averages = {k: "" for k in CLASS_KEYS}
            avg_total = ""
            avg_moving = avg_slow = avg_stop = avg_stop_ratio = ""
            avg_smp = avg_occ = avg_px_occ = ""
            est_flow_veh = est_q_smp = ""
            avg_dj_fl = avg_dj_sp = avg_dj_ef = max_dj_ef = ""
            lancar_pct = padat_pct = macet_pct = ""
            dominant_cond = ""
            dominant_emp = ""
            avg_infer_ms = ""
            proc_fps = 0.0

        return {
            "sampled_frames": n,
            "dominant_emp_mode": dominant_emp,
            "average_active_motor": averages["motor"],
            "average_active_mobil": averages["mobil"],
            "average_active_bus": averages["bus"],
            "average_active_truk": averages["truk"],
            "average_active_total": avg_total,
            "max_active_total": self.max_total if n else "",
            "avg_smp_frame": avg_smp,
            "c_roi_capacity_smp": round(ROI_MAX_SMP_CAPACITY, 2),
            "avg_roi_occupancy_pct": avg_occ,
            "avg_pixel_occupancy_pct": avg_px_occ,
            "est_flow_veh_per_hour": est_flow_veh,
            "est_q_smp_per_hour": est_q_smp,
            "road_capacity_c_smp_per_hour": round(ROAD_CAPACITY_SMP_PER_HOUR, 1),
            "avg_dj_flow": avg_dj_fl,
            "avg_dj_spatial": avg_dj_sp,
            "avg_dj_effective": avg_dj_ef,
            "max_dj_effective": max_dj_ef,
            "avg_moving_vehicles": avg_moving,
            "avg_slow_vehicles": avg_slow,
            "avg_stopped_vehicles": avg_stop,
            "avg_stopped_ratio": avg_stop_ratio,
            "pct_time_lancar": lancar_pct,
            "pct_time_padat": padat_pct,
            "pct_time_macet": macet_pct,
            "dominant_cond": dominant_cond,
            "average_inference_ms": avg_infer_ms,
            "processing_fps": proc_fps,
        }


# ============================================================
# 7. HELPER PARSING FILENAME, FILE VIDEO, ROI & CSV
# ============================================================
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


def resolve_model_path(cli_model: Optional[Path]) -> Path:
    if cli_model is not None:
        if not cli_model.is_file():
            raise FileNotFoundError(f"File model ONNX tidak ditemukan: {cli_model}")
        return cli_model

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    alt_model = BASE_DIR / "models" / "v-1-1.onnx"
    if alt_model.is_file():
        print(
            f"[INFO] {DEFAULT_MODEL_PATH.name} tidak ditemukan, menggunakan {alt_model.name}",
            flush=True,
        )
        return alt_model

    raise FileNotFoundError(
        f"Model ONNX tidak ditemukan di {DEFAULT_MODEL_PATH} maupun {alt_model}"
    )


def load_roi(config_path: Path, frame_width: int, frame_height: int) -> np.ndarray:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "points" not in data:
        raise ValueError("roi_config.json harus memiliki key 'points'.")

    points = np.asarray(data["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Format ROI tidak valid. Minimal 3 titik [[x1,y1], ...].")

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
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


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


# ============================================================
# 8. EKSTRAKSI 1 FILE VIDEO (RINGKASAN VIDEO + BUCKET 1 MENIT)
# ============================================================
def process_video_file(
    video_path: Path,
    detector: PureONNXYOLO,
    roi_config_path: Path,
    inference_interval: float,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    sub_file,
    sub_writer: csv.DictWriter,
    vid_file,
    vid_writer: csv.DictWriter,
) -> List[dict]:
    location, video_start_dt = parse_video_metadata(video_path)
    print("\n" + "=" * 75, flush=True)
    print(f"Memproses Video : {video_path.name}", flush=True)
    print(f"Lokasi          : {location}", flush=True)
    print(f"Waktu Mulai WIB : {video_start_dt.isoformat(timespec='seconds')}", flush=True)
    print("=" * 75, flush=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        err_msg = f"Gagal membuka file video: {video_path}"
        print(f"[ERROR] {err_msg}", flush=True)
        empty_vid_row = {k: "" for k in VIDEO_SUMMARY_CSV_FIELDS}
        empty_vid_row.update(
            {
                "video_file": video_path.name,
                "location": location,
                "video_start_local": timestamp_local(video_start_dt),
                "sampled_frames": 0,
                "c_roi_capacity_smp": round(ROI_MAX_SMP_CAPACITY, 2),
                "road_capacity_c_smp_per_hour": round(ROAD_CAPACITY_SMP_PER_HOUR, 1),
                "processing_fps": 0.0,
                "status": "FAILED",
                "error_message": err_msg,
            }
        )
        append_csv_row(vid_file, vid_writer, empty_vid_row)
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0 or math.isnan(fps):
        fps = 25.0
        print("[WARN] FPS video tidak terbaca; menggunakan fallback 25.0 FPS.", flush=True)

    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_duration_sec = (total_frames_est / fps) if total_frames_est > 0 else 0.0
    print(
        f"[INFO] FPS Video={fps:.2f} | Estimasi Frame={total_frames_est} | "
        f"Estimasi Durasi={est_duration_sec:.1f}s | Interval Inferensi={inference_interval:.2f}s",
        flush=True,
    )

    tracker = SimpleCentroidTracker()
    analyzer = VehicleMotionAnalyzer(observation_window=OBSERVATION_WINDOW_SEC)
    pkji_calc = PKJIFlowAndOccupancyCalculator(rolling_window_sec=FLOW_ROLLING_WINDOW_SEC)

    smooth_maxlen = max(
        1, int(round(TRAFFIC_STATE_SMOOTH_SEC / max(0.05, inference_interval)))
    )
    recent_traffic_states: deque = deque(maxlen=smooth_maxlen)

    video_acc = MetricsAccumulator(inference_interval=inference_interval)
    current_minute_bucket: Optional[datetime] = None
    minute_acc: Optional[MetricsAccumulator] = None
    minute_rows: List[dict] = []

    roi_polygon: Optional[np.ndarray] = None
    roi_pixel_area = 1.0

    frame_idx = 0
    last_sampled_video_sec: Optional[float] = None
    last_cleanup_sec = -1
    last_progress_wall = time.monotonic()

    def flush_minute_bucket(bucket_dt: datetime, acc: MetricsAccumulator) -> dict:
        stats = acc.summarize()
        dominant_cond = stats.pop("dominant_cond")
        actual_dur = acc.video_span_seconds()
        hour_bucket = bucket_dt.replace(minute=0, second=0, microsecond=0)

        if STOP_REQUESTED:
            st = "INTERRUPTED"
        elif acc.sampled_frames == 0:
            st = "FAILED"
        elif actual_dur < 45.0 or acc.sampled_frames < 30:
            st = "PARTIAL"
        else:
            st = "OK"

        row = {
            "video_file": video_path.name,
            "location": location,
            "hour_bucket": hour_bucket.isoformat(timespec="seconds"),
            "one_min_bucket": bucket_dt.isoformat(timespec="seconds"),
            "session_point": bucket_dt.minute + 1,
            "sample_start_local": timestamp_local(acc.first_dt),
            "sample_end_local": timestamp_local(acc.last_dt),
            "target_sample_duration_seconds": 60,
            "actual_sample_duration_seconds": actual_dur,
            **stats,
            "dominant_traffic_condition": dominant_cond,
            "one_min_traffic_condition": dominant_cond,
            "status": st,
            "error_message": summarize_error_messages(acc.errors),
        }
        append_csv_row(sub_file, sub_writer, row)
        print(
            f"  [1-MIN] {bucket_dt.strftime('%H:%M')} | dur={actual_dur:.1f}s | "
            f"frames={acc.sampled_frames} | AvgKend={stats['average_active_total']} | "
            f"Q={stats['est_q_smp_per_hour']} | DJ={stats['avg_dj_effective']} "
            f"(max={stats['max_dj_effective']}) | {stats['average_inference_ms']}ms | {dominant_cond}",
            flush=True,
        )
        return row

    try:
        while not STOP_REQUESTED:
            grabbed = cap.grab()
            if not grabbed:
                break

            video_sec = frame_idx / fps
            frame_idx += 1

            if (
                last_sampled_video_sec is not None
                and (video_sec - last_sampled_video_sec) < (inference_interval - 1e-6)
            ):
                continue

            retrieved, frame = cap.retrieve()
            if not retrieved or frame is None:
                err_txt = "Frame gagal di-decode"
                video_acc.add_error(err_txt)
                if minute_acc is not None:
                    minute_acc.add_error(err_txt)
                continue

            last_sampled_video_sec = video_sec
            frame_dt = video_start_dt + timedelta(seconds=video_sec)

            if roi_polygon is None:
                frame_height, frame_width = frame.shape[:2]
                roi_polygon = load_roi(roi_config_path, frame_width, frame_height)
                roi_pixel_area = max(1.0, float(cv2.contourArea(roi_polygon)))
                print(
                    f"[INFO] Resolusi Video: {frame_width}x{frame_height} | "
                    f"Area ROI: {roi_pixel_area:.0f} px",
                    flush=True,
                )

            sec_int = int(video_sec)
            if sec_int != last_cleanup_sec:
                analyzer.cleanup_stale(video_sec)
                last_cleanup_sec = sec_int

            # 1. Inferensi ONNX (CUDA GPU)
            infer_start = time.perf_counter()
            raw_detections = detector.infer(
                frame=frame,
                imgsz=imgsz,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=MAX_DETECTIONS,
            )
            inference_ms = (time.perf_counter() - infer_start) * 1000.0

            # 2. Filter kendaraan dalam poligon ROI
            roi_candidates = []
            for x1, y1, x2, y2, _conf, class_id in raw_detections:
                raw_name = detector.names.get(class_id, str(class_id))
                v_group = map_vehicle_class(raw_name)
                if v_group is None:
                    continue

                anchor_x = int(round((x1 + x2) / 2.0))
                anchor_y = int(round(y2))
                inside = (
                    cv2.pointPolygonTest(
                        roi_polygon, (float(anchor_x), float(anchor_y)), False
                    )
                    >= 0
                )
                if not inside:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                roi_candidates.append((x1, y1, x2, y2, cx, cy, v_group))

            # 3. Centroid Tracking & Analisis Gerak (berbasis video_sec)
            centroids = [(item[4], item[5]) for item in roi_candidates]
            track_ids = tracker.update(centroids)

            frame_counts = new_counts()
            motion_counts = {"Moving": 0, "SLOW": 0, "STOP": 0}
            active_tracks_in_frame: List[Tuple[int, str, float]] = []
            frame_bbox_area_sum = 0.0

            for candidate, track_id in zip(roi_candidates, track_ids):
                x1, y1, x2, y2, cx, cy, v_group = candidate
                motion_info = analyzer.update_vehicle(
                    int(track_id), video_sec, int(round(cx)), int(round(cy))
                )
                m_state = str(motion_info["motion_state"])
                dwell_t = float(motion_info["dwell_time"])

                frame_counts[v_group] += 1
                motion_counts[m_state] += 1
                active_tracks_in_frame.append((int(track_id), v_group, dwell_t))
                frame_bbox_area_sum += max(0.0, (x2 - x1) * (y2 - y1))

            # 4. Hitung Metrik PKJI 2023 & Kondisi Lalu Lintas
            pkji_calc.register_active_tracks(video_sec, active_tracks_in_frame)
            pkji_metrics = pkji_calc.compute_pkji_metrics(video_sec, frame_counts)

            roi_occupancy_pct = float(pkji_metrics["roi_occupancy_pct"])
            dj_effective = float(pkji_metrics["dj_effective"])
            pixel_occupancy_pct = min(
                100.0, (frame_bbox_area_sum / roi_pixel_area) * 100.0
            )

            total_in_roi = sum(frame_counts.values())
            instant_status = classify_traffic_condition(
                roi_occupancy_pct,
                dj_effective,
                total_in_roi,
                motion_counts["Moving"],
                motion_counts["SLOW"],
                motion_counts["STOP"],
            )
            recent_traffic_states.append(instant_status)
            smoothed_status = Counter(recent_traffic_states).most_common(1)[0][0]

            # 5. Masukkan ke akumulator video utuh & akumulator 1 menit
            video_acc.add_frame(
                video_sec=video_sec,
                frame_dt=frame_dt,
                frame_counts=frame_counts,
                motion_counts=motion_counts,
                pkji_metrics=pkji_metrics,
                pixel_occupancy_pct=pixel_occupancy_pct,
                smoothed_status=smoothed_status,
                inference_ms=inference_ms,
            )

            frame_min_bucket = frame_dt.replace(second=0, microsecond=0)
            if current_minute_bucket is None:
                current_minute_bucket = frame_min_bucket
                minute_acc = MetricsAccumulator(inference_interval=inference_interval)
            elif frame_min_bucket != current_minute_bucket and minute_acc is not None:
                minute_rows.append(flush_minute_bucket(current_minute_bucket, minute_acc))
                current_minute_bucket = frame_min_bucket
                minute_acc = MetricsAccumulator(inference_interval=inference_interval)

            if minute_acc is not None:
                minute_acc.add_frame(
                    video_sec=video_sec,
                    frame_dt=frame_dt,
                    frame_counts=frame_counts,
                    motion_counts=motion_counts,
                    pkji_metrics=pkji_metrics,
                    pixel_occupancy_pct=pixel_occupancy_pct,
                    smoothed_status=smoothed_status,
                    inference_ms=inference_ms,
                )

            now_wall = time.monotonic()
            if now_wall - last_progress_wall >= 15.0:
                pct_str = (
                    f"{(frame_idx / total_frames_est * 100.0):.1f}%"
                    if total_frames_est > 0
                    else f"{video_sec:.1f}s"
                )
                print(
                    f"  [PROGRESS] {pct_str} | video_t={video_sec:.1f}s | "
                    f"sampled={video_acc.sampled_frames} | "
                    f"DJ={dj_effective:.3f} | Status={smoothed_status}",
                    flush=True,
                )
                last_progress_wall = now_wall

    except KeyboardInterrupt:
        request_stop(signal.SIGINT, None)
    except Exception as exc:
        err_txt = f"{type(exc).__name__}: {exc}"
        video_acc.add_error(err_txt)
        if minute_acc is not None:
            minute_acc.add_error(err_txt)
        print(f"[ERROR] {err_txt}", flush=True)
    finally:
        cap.release()
        gc.collect()

    # Flush sisa bucket 1 menit terakhir pada video
    if (
        current_minute_bucket is not None
        and minute_acc is not None
        and minute_acc.sampled_frames > 0
    ):
        minute_rows.append(flush_minute_bucket(current_minute_bucket, minute_acc))

    # Tulis ringkasan utuh 1 file video
    actual_video_duration = round(frame_idx / fps, 3) if frame_idx > 0 else 0.0
    video_end_dt = video_start_dt + timedelta(seconds=actual_video_duration)
    v_stats = video_acc.summarize()
    video_cond = v_stats.pop("dominant_cond")

    if STOP_REQUESTED:
        vid_status = "INTERRUPTED"
    elif video_acc.sampled_frames == 0:
        vid_status = "FAILED"
    else:
        vid_status = "OK"

    video_row = {
        "video_file": video_path.name,
        "location": location,
        "video_start_local": timestamp_local(video_start_dt),
        "video_end_local": timestamp_local(video_end_dt),
        "video_duration_seconds": actual_video_duration,
        "video_fps": round(fps, 3),
        **v_stats,
        "video_traffic_condition": video_cond,
        "status": vid_status,
        "error_message": summarize_error_messages(video_acc.errors),
    }
    append_csv_row(vid_file, vid_writer, video_row)

    print(
        f"[VIDEO SAVED] {video_path.name} | Durasi={actual_video_duration:.1f}s | "
        f"Frames={video_acc.sampled_frames} | Infer={v_stats['average_inference_ms']}ms | "
        f"AvgKend={v_stats['average_active_total']} | Okupansi={v_stats['avg_roi_occupancy_pct']}% | "
        f"Q={v_stats['est_q_smp_per_hour']} SMP/jam | "
        f"DJ_Eff={v_stats['avg_dj_effective']} (Max={v_stats['max_dj_effective']}) -> {video_cond}",
        flush=True,
    )
    return minute_rows


# ============================================================
# 9. AGREGASI MULTI-INTERVAL (5, 15, 20, 25, 30, 35, 40, 50, 60 MENIT)
# ============================================================
def compute_aggregate_metrics(
    sessions: List[dict], expected_sessions: int
) -> Dict[str, object]:
    valid_sessions = [s for s in sessions if int(s["sampled_frames"] or 0) > 0]
    total_target_sec = sum(int(s["target_sample_duration_seconds"]) for s in sessions)
    total_actual_sec = round(
        sum(float(s["actual_sample_duration_seconds"]) for s in sessions), 3
    )
    total_frames = sum(int(s["sampled_frames"]) for s in sessions)

    first_sample_start = next(
        (s["sample_start_local"] for s in sessions if s.get("sample_start_local")),
        "",
    )
    last_sample_end = next(
        (s["sample_end_local"] for s in reversed(sessions) if s.get("sample_end_local")),
        "",
    )

    if valid_sessions:
        n_valid = len(valid_sessions)

        def mean_metric(col: str, ndigits: int = 3) -> float:
            return round(
                sum(float(s[col]) for s in valid_sessions) / n_valid, ndigits
            )

        avg_by_class = {
            key: mean_metric(f"average_active_{key}", 3) for key in CLASS_KEYS
        }
        avg_total = mean_metric("average_active_total", 3)
        max_total = max(int(s["max_active_total"]) for s in valid_sessions)

        avg_smp = mean_metric("avg_smp_frame", 3)
        avg_occ = mean_metric("avg_roi_occupancy_pct", 2)
        avg_px_occ = mean_metric("avg_pixel_occupancy_pct", 2)
        est_flow_veh = mean_metric("est_flow_veh_per_hour", 1)
        est_q_smp = mean_metric("est_q_smp_per_hour", 1)
        avg_dj_fl = mean_metric("avg_dj_flow", 3)
        avg_dj_sp = mean_metric("avg_dj_spatial", 3)
        avg_dj_ef = mean_metric("avg_dj_effective", 3)
        max_dj_ef = round(
            max(float(s["max_dj_effective"]) for s in valid_sessions), 3
        )

        avg_moving = mean_metric("avg_moving_vehicles", 3)
        avg_slow = mean_metric("avg_slow_vehicles", 3)
        avg_stop = mean_metric("avg_stopped_vehicles", 3)
        avg_stop_ratio = round(avg_stop / avg_total, 3) if avg_total > 0 else 0.0

        pct_lancar = mean_metric("pct_time_lancar", 2)
        pct_padat = mean_metric("pct_time_padat", 2)
        pct_macet = mean_metric("pct_time_macet", 2)

        if (
            est_flow_veh >= EMP_SWITCH_FLOW_VEH_PER_HOUR
            or avg_occ >= INITIAL_OCCUPANCY_THRESHOLD_FOR_HIGH_EMP_PCT
        ):
            dominant_emp = "HIGH_FLOW (>=1050/lajur)"
        else:
            dominant_emp = "LOW_FLOW (<1050/lajur)"

        status_pcts = [
            ("Macet (Congestion)", pct_macet),
            ("Padat (Slow Traffic)", pct_padat),
            ("Lancar (Free Flow)", pct_lancar),
        ]
        dominant_by_time = max(status_pcts, key=lambda x: x[1])[0]
        metric_condition = classify_traffic_condition(
            avg_occ, avg_dj_ef, avg_total, avg_moving, avg_slow, avg_stop
        )
        agg_condition = (
            dominant_by_time
            if max(pct_lancar, pct_padat, pct_macet) >= 50.0
            else metric_condition
        )

        avg_infer_ms = round(
            sum(
                float(s["average_inference_ms"]) * int(s["sampled_frames"])
                for s in valid_sessions
            )
            / total_frames,
            3,
        )
        proc_fps = (
            round(
                sum(
                    float(s["processing_fps"]) * int(s["sampled_frames"])
                    for s in valid_sessions
                )
                / total_frames,
                3,
            )
            if total_frames > 0
            else 0.0
        )
    else:
        avg_by_class = {key: "" for key in CLASS_KEYS}
        avg_total = max_total = ""
        avg_smp = avg_occ = avg_px_occ = ""
        est_flow_veh = est_q_smp = ""
        avg_dj_fl = avg_dj_sp = avg_dj_ef = max_dj_ef = ""
        avg_moving = avg_slow = avg_stop = avg_stop_ratio = ""
        pct_lancar = pct_padat = pct_macet = ""
        dominant_emp = ""
        agg_condition = ""
        avg_infer_ms = ""
        proc_fps = 0.0

    statuses = [s["status"] for s in sessions]
    if len(valid_sessions) >= expected_sessions and all(st == "OK" for st in statuses):
        agg_status = "OK"
    elif len(valid_sessions) == 0:
        agg_status = "FAILED"
    elif "INTERRUPTED" in statuses:
        agg_status = "INTERRUPTED"
    else:
        agg_status = "PARTIAL"

    # Ringkas error_message dengan Counter agar tidak berulang panjang di CSV
    error_summary = summarize_error_messages(
        s["error_message"] for s in sessions if s.get("error_message")
    )

    return {
        "sessions_completed": len(valid_sessions),
        "first_sample_start_local": first_sample_start,
        "last_sample_end_local": last_sample_end,
        "total_target_duration_seconds": total_target_sec,
        "total_actual_duration_seconds": total_actual_sec,
        "total_sampled_frames": total_frames,
        "dominant_emp_mode": dominant_emp,
        "average_active_motor": avg_by_class["motor"],
        "average_active_mobil": avg_by_class["mobil"],
        "average_active_bus": avg_by_class["bus"],
        "average_active_truk": avg_by_class["truk"],
        "average_active_total": avg_total,
        "max_active_total": max_total,
        "avg_smp_frame": avg_smp,
        "c_roi_capacity_smp": round(ROI_MAX_SMP_CAPACITY, 2),
        "avg_roi_occupancy_pct": avg_occ,
        "avg_pixel_occupancy_pct": avg_px_occ,
        "est_flow_veh_per_hour": est_flow_veh,
        "est_q_smp_per_hour": est_q_smp,
        "road_capacity_c_smp_per_hour": round(ROAD_CAPACITY_SMP_PER_HOUR, 1),
        "avg_dj_flow": avg_dj_fl,
        "avg_dj_spatial": avg_dj_sp,
        "avg_dj_effective": avg_dj_ef,
        "max_dj_effective": max_dj_ef,
        "avg_moving_vehicles": avg_moving,
        "avg_slow_vehicles": avg_slow,
        "avg_stopped_vehicles": avg_stop,
        "avg_stopped_ratio": avg_stop_ratio,
        "pct_time_lancar": pct_lancar,
        "pct_time_padat": pct_padat,
        "pct_time_macet": pct_macet,
        "agg_condition": agg_condition,
        "average_inference_ms": avg_infer_ms,
        "processing_fps": proc_fps,
        "status": agg_status,
        "error_message": error_summary,
    }


def get_day_window_bounds(
    dt: datetime, interval_minutes: int
) -> Tuple[datetime, datetime, int]:
    """Menentukan batas window sejajar 00:00 hari yang sama."""
    day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    minutes_since_midnight = int((dt - day_start).total_seconds() // 60)
    bucket_idx = minutes_since_midnight // interval_minutes
    win_start = day_start + timedelta(minutes=bucket_idx * interval_minutes)
    win_end = min(win_start + timedelta(minutes=interval_minutes), day_end)
    expected_sessions = max(1, int((win_end - win_start).total_seconds() // 60))
    return win_start, win_end, expected_sessions


def write_multi_interval_summaries(
    all_minute_rows: List[dict],
    output_dir: Path,
    append: bool = False,
) -> None:
    if not all_minute_rows:
        return

    for interval_m in ALL_AGGREGATION_INTERVALS:
        grouped: Dict[datetime, Tuple[datetime, int, List[dict]]] = {}
        for row in all_minute_rows:
            bucket_dt = datetime.fromisoformat(row["one_min_bucket"])
            w_start, w_end, expected_n = get_day_window_bounds(bucket_dt, interval_m)
            if w_start not in grouped:
                grouped[w_start] = (w_end, expected_n, [])
            grouped[w_start][2].append(row)

        if interval_m == 60:
            csv_path = output_dir / "hourly_pkji2_summary.csv"
            fields = HOURLY_CSV_FIELDS
        else:
            csv_path = output_dir / f"summary_{interval_m}min_pkji2.csv"
            fields = INTERVAL_CSV_FIELDS

        f_handle, writer = open_csv_writer(csv_path, fields, append=append)
        try:
            for w_start in sorted(grouped.keys()):
                w_end, expected_n, sessions = grouped[w_start]
                metrics = compute_aggregate_metrics(sessions, expected_n)
                agg_condition = metrics.pop("agg_condition")

                if interval_m == 60:
                    out_row = {
                        "scheduled_hour": w_start.isoformat(timespec="seconds"),
                        **metrics,
                        "hourly_traffic_condition": agg_condition,
                    }
                else:
                    out_row = {
                        "interval_minutes": interval_m,
                        "window_start": w_start.isoformat(timespec="seconds"),
                        "window_end": w_end.isoformat(timespec="seconds"),
                        "expected_sessions": expected_n,
                        **metrics,
                        "window_traffic_condition": agg_condition,
                    }
                append_csv_row(f_handle, writer, out_row)
        finally:
            f_handle.close()


# ============================================================
# 10. CLI ENTRYPOINT
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ekstraksi DJ & Metrik PKJI 2023 (ONNX CUDA GPU) dari File Video Rekaman"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path ke 1 file video (.mp4) atau folder berisi kumpulan video rekaman.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Cari file video hingga ke subfolder jika --input berupa folder.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"Path model ONNX (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=DEFAULT_ROI_CONFIG_PATH,
        help=f"Path file roi_config.json (default: {DEFAULT_ROI_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Folder output CSV (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "auto", "cpu"],
        default="cuda",
        help="Target eksekusi ONNX Runtime: cuda (wajib GPU), auto (GPU lalu CPU), atau cpu (default: cuda).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Indeks GPU CUDA yang digunakan (default: 0).",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=DEFAULT_INFERENCE_INTERVAL_SECONDS,
        help=f"Interval sampling waktu video dalam detik (default: {DEFAULT_INFERENCE_INTERVAL_SECONDS}s / ~5 FPS).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Ukuran input gambar ONNX (default: {DEFAULT_IMGSZ}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold deteksi (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IOU threshold NMS (default: {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help=f"Jumlah CPU thread jika memakai CPUExecutionProvider (default: {DEFAULT_CPU_THREADS}).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Tambahkan baris ke CSV yang sudah ada (default: buat baru / timpa).",
    )
    parser.add_argument(
        "--no-multi-interval",
        action="store_true",
        help="Lewati pembuatan file agregasi 5-60 menit (hanya simpan per video & 1 menit).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.inference_interval <= 0:
        raise ValueError("--inference-interval harus lebih besar dari 0.")
    if not args.roi.is_file():
        raise FileNotFoundError(f"File ROI tidak ditemukan: {args.roi}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    model_path = resolve_model_path(args.model)
    video_files = collect_video_files(args.input, recursive=args.recursive)
    if not video_files:
        raise FileNotFoundError(f"Tidak ada file video ditemukan di: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_csv_path = args.output_dir / "video_summary_pkji2.csv"
    subsession_csv_path = args.output_dir / "subsession_1min_pkji2.csv"

    print("=" * 75, flush=True)
    print("PURE ONNX (CUDA GPU) - PKJI 2023 (MODE 2) OFFLINE VIDEO DJ EXTRACTOR", flush=True)
    print("=" * 75, flush=True)
    print(f"Jumlah Video         : {len(video_files)}", flush=True)
    print(f"Model ONNX           : {model_path}", flush=True)
    print(f"ROI Config           : {args.roi}", flush=True)
    print(f"Output Dir           : {args.output_dir}", flush=True)
    print(f"Output Per Video     : {video_csv_path}", flush=True)
    print(f"Output Per 1 Menit   : {subsession_csv_path}", flush=True)
    print(
        f"Kapasitas Jalan (C)  : {ROAD_CAPACITY_SMP_PER_HOUR:.1f} SMP/jam "
        f"({ACTIVE_LANES_IN_ROI} lajur, FC_LJ={PKJI_FC_LJ})",
        flush=True,
    )
    print(f"Kapasitas Ruang ROI  : {ROI_MAX_SMP_CAPACITY:.2f} SMP", flush=True)

    detector = PureONNXYOLO(
        model_path=model_path,
        device=args.device,
        gpu_id=args.gpu_id,
        cpu_threads=args.threads,
        warmup_imgsz=args.imgsz,
    )

    vid_file, vid_writer = open_csv_writer(
        video_csv_path, VIDEO_SUMMARY_CSV_FIELDS, append=args.append
    )
    sub_file, sub_writer = open_csv_writer(
        subsession_csv_path, SUBSESSION_CSV_FIELDS, append=args.append
    )

    all_minute_rows: List[dict] = []
    try:
        for idx, video_path in enumerate(video_files, start=1):
            if STOP_REQUESTED:
                break
            print(f"\n>>> [{idx}/{len(video_files)}] {video_path.name}", flush=True)
            rows = process_video_file(
                video_path=video_path,
                detector=detector,
                roi_config_path=args.roi,
                inference_interval=args.inference_interval,
                imgsz=args.imgsz,
                conf_thres=args.conf,
                iou_thres=args.iou,
                sub_file=sub_file,
                sub_writer=sub_writer,
                vid_file=vid_file,
                vid_writer=vid_writer,
            )
            all_minute_rows.extend(rows)
    finally:
        vid_file.close()
        sub_file.close()

    if not args.no_multi_interval and all_minute_rows:
        write_multi_interval_summaries(
            all_minute_rows=all_minute_rows,
            output_dir=args.output_dir,
            append=args.append,
        )
        print(
            f"\n[INFO] Agregasi multi-interval (5..60 menit) tersimpan di {args.output_dir}",
            flush=True,
        )

    print("\nEkstraksi selesai. Seluruh file CSV telah disimpan dengan aman.", flush=True)
    return 0 if not STOP_REQUESTED else 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise