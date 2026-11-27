"""Tracker: centroid lama (--legacy), prediksi gerak (default), dan penghitung garis."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


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


class MotionTracker:
    """Tracker untuk sampling rendah (default mode): prediksi kecepatan konstan.

    Pada 5 sampel/detik kotak kendaraan sering nyaris tidak beririsan antar
    sampel, sehingga pencocokan IoU (ByteTrack) gagal. Di sini pasangan dipilih
    (Hungarian) dari jarak titik tengah yang dinormalisasi ukuran kotak, IoU, dan
    perubahan luas. Setiap deteksi selalu mendapat ID.
    """

    MIN_IOU = 0.3          # tumpang tindih sebesar ini selalu lolos gerbang
    MAX_LOG_AREA = 1.4     # luas tidak boleh berubah lebih dari sekitar 4x
    SIZE_WEIGHT = 0.5
    VELOCITY_ALPHA = 0.5   # bobot pengukuran kecepatan terbaru

    def __init__(self, max_age_sec: float = 0.8, gate: float = 0.6, new_gate: float = 1.0) -> None:
        self.max_age_sec = max_age_sec
        self.gate = gate          # track yang kecepatannya sudah diketahui
        self.new_gate = new_gate  # track baru satu pengamatan
        self.next_id = 1
        self.tracks: Dict[int, list] = {}  # id -> [cx, cy, w, h, vx, vy, t, jumlah]

    def update(self, dets: np.ndarray, t: float) -> List[int]:
        """dets (N,>=4) x1,y1,x2,y2,... pada waktu t (detik) -> ID per baris."""
        self.tracks = {
            k: v for k, v in self.tracks.items() if t - v[6] <= self.max_age_sec + 1e-6
        }
        ids: List[Optional[int]] = [None] * len(dets)
        if not len(dets):
            return ids
        boxes = np.asarray(dets, dtype=np.float64)[:, :4]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1.0)
        if self.tracks:
            self._match(ids, centers, sizes, t)
        for j in range(len(ids)):
            if ids[j] is None:
                ids[j] = self.next_id
                self.next_id += 1
                self.tracks[ids[j]] = [*centers[j], *sizes[j], 0.0, 0.0, t, 1]
                continue
            cx, cy, _, _, vx, vy, last, seen = self.tracks[ids[j]]
            dt = max(t - last, 1e-3)
            new_vx, new_vy = (centers[j, 0] - cx) / dt, (centers[j, 1] - cy) / dt
            if seen >= 2:
                alpha = self.VELOCITY_ALPHA
                new_vx = alpha * new_vx + (1.0 - alpha) * vx
                new_vy = alpha * new_vy + (1.0 - alpha) * vy
            self.tracks[ids[j]] = [*centers[j], *sizes[j], new_vx, new_vy, t, seen + 1]
        return ids

    def _match(self, ids: list, centers: np.ndarray, sizes: np.ndarray, t: float) -> None:
        keys = list(self.tracks)
        state = np.array([self.tracks[k] for k in keys])
        predicted = state[:, :2] + state[:, 4:6] * (t - state[:, 6:7])
        area_t = np.prod(state[:, 2:4], axis=1)[:, None]
        area_d = np.prod(sizes, axis=1)[None]
        scale = (area_t * area_d) ** 0.25
        distance = np.linalg.norm(predicted[:, None] - centers[None], axis=2) / scale
        low = np.maximum((predicted - state[:, 2:4] / 2)[:, None], (centers - sizes / 2)[None])
        high = np.minimum((predicted + state[:, 2:4] / 2)[:, None], (centers + sizes / 2)[None])
        inter = np.prod(np.clip(high - low, 0, None), axis=2)
        overlap = inter / np.maximum(area_t + area_d - inter, 1e-9)
        area_change = np.abs(np.log(area_d / area_t))

        gate = np.where(state[:, 7] >= 2, self.gate, self.new_gate)[:, None]
        valid = ((distance <= gate) | (overlap >= self.MIN_IOU)) & (area_change < self.MAX_LOG_AREA)
        cost = distance + (1.0 - overlap) + self.SIZE_WEIGHT * area_change
        cost = np.where(valid, cost, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if valid[i, j]:
                ids[j] = keys[i]


def _side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_cross(p1, p2, q1, q2) -> bool:
    """True bila lintasan p1->p2 memotong garis hitung q1-q2."""
    d1, d2 = _side(q1, q2, p1), _side(q1, q2, p2)
    d3, d4 = _side(p1, p2, q1), _side(p1, p2, q2)
    # titik tepat di garis dihitung milik sisi positif supaya tidak terhitung dua kali
    return (d1 >= 0) != (d2 >= 0) and d3 * d4 <= 0


class LineCounter:
    """Mencatat track yang titik bawah-tengahnya melintasi garis hitung (sekali per track)."""

    def __init__(self, line: np.ndarray) -> None:
        self.a, self.b = (tuple(map(float, p)) for p in line)
        self.last: Dict[int, Tuple[float, float]] = {}
        self.counted: set = set()

    def update(self, track_id: int, anchor: Tuple[float, float]) -> bool:
        prev = self.last.get(track_id)
        self.last[track_id] = anchor
        if prev is None or track_id in self.counted:
            return False
        if segments_cross(prev, anchor, self.a, self.b):
            self.counted.add(track_id)
            return True
        return False
