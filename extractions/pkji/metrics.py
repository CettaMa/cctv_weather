"""Perhitungan PKJI 2023, analisis gerak, dan klasifikasi kondisi lalu lintas."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Set, Tuple

from .common import new_counts
from .config import CLASS_KEYS, PKJIParams


class PKJIFlowAndOccupancyCalculator:
    """Menghitung metrik Standar PKJI 2023 (EMP, SMP, Okupansi ROI, Q, dan DJ)."""

    def __init__(self, params: PKJIParams) -> None:
        self.p = params
        self.rolling_window_sec = params.flow_rolling_window_sec
        self.counted_track_ids: Set[int] = set()
        self.rolling_departures: deque = deque()

    def register_active_tracks(
        self, timestamp_sec: float, active_tracks: List[Tuple[int, str, float]]
    ) -> None:
        for track_id, v_class, dwell_time in active_tracks:
            if (
                track_id not in self.counted_track_ids
                and dwell_time >= self.p.min_dwell_for_valid_flow_sec
            ):
                self.counted_track_ids.add(track_id)
                self.rolling_departures.append((timestamp_sec, track_id, v_class))

        self._expire(timestamp_sec)

    def exclude_track(self, track_id: int) -> None:
        """Kendaraan yang sudah ada saat pengamatan dimulai tidak dihitung sebagai arus."""
        self.counted_track_ids.add(track_id)

    def register_departure(self, timestamp_sec: float, track_id: int, v_class: str) -> None:
        """Satu kejadian melintasi garis hitung."""
        self.rolling_departures.append((timestamp_sec, track_id, v_class))

    def _expire(self, timestamp_sec: float) -> None:
        while (
            self.rolling_departures
            and (timestamp_sec - self.rolling_departures[0][0]) > self.rolling_window_sec
        ):
            self.rolling_departures.popleft()

    def compute_pkji_metrics(
        self, timestamp_sec: float, frame_counts: Dict[str, int]
    ) -> Dict[str, float | str]:
        effective_window_sec = min(
            max(self.p.min_flow_extrapolation_window_sec, timestamp_sec),
            self.rolling_window_sec,
        )
        self._expire(timestamp_sec)
        window_counts = new_counts()
        for _, _, v_class in self.rolling_departures:
            window_counts[v_class] += 1

        scale_to_hour = 3600.0 / effective_window_sec
        hourly_counts = {k: window_counts[k] * scale_to_hour for k in CLASS_KEYS}
        total_flow_veh_per_hour = sum(hourly_counts.values())

        smp_frame_low = sum(frame_counts[k] * self.p.emp_low_flow[k] for k in CLASS_KEYS)
        occ_low_pct = (smp_frame_low / self.p.roi_max_smp_capacity) * 100.0

        if (
            total_flow_veh_per_hour >= self.p.emp_switch_flow_veh_per_hour
            or occ_low_pct >= self.p.initial_occupancy_threshold_for_high_emp_pct
        ):
            active_emp = self.p.emp_high_flow
            emp_mode = "HIGH_FLOW (>=1050/lajur)"
        else:
            active_emp = self.p.emp_low_flow
            emp_mode = "LOW_FLOW (<1050/lajur)"

        smp_frame = sum(frame_counts[k] * active_emp[k] for k in CLASS_KEYS)
        roi_occupancy_pct = min(100.0, (smp_frame / self.p.roi_max_smp_capacity) * 100.0)

        q_flow_smp_per_hour = sum(hourly_counts[k] * active_emp[k] for k in CLASS_KEYS)
        dj_from_flow = q_flow_smp_per_hour / self.p.road_capacity_smp_per_hour

        if roi_occupancy_pct <= self.p.occupancy_padat_pct:
            dj_spatial = (roi_occupancy_pct / self.p.occupancy_padat_pct) * self.p.dj_padat_threshold
        elif roi_occupancy_pct <= self.p.occupancy_macet_pct:
            ratio_mid = (roi_occupancy_pct - self.p.occupancy_padat_pct) / (
                self.p.occupancy_macet_pct - self.p.occupancy_padat_pct
            )
            dj_spatial = self.p.dj_padat_threshold + ratio_mid * (
                self.p.dj_macet_threshold - self.p.dj_padat_threshold
            )
        else:
            ratio_high = (roi_occupancy_pct - self.p.occupancy_macet_pct) / (
                100.0 - self.p.occupancy_macet_pct
            )
            dj_spatial = self.p.dj_macet_threshold + ratio_high * (1.00 - self.p.dj_macet_threshold)

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

    def __init__(self, params: PKJIParams) -> None:
        self.p = params
        self.observation_window = params.observation_window_sec
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
            dwell_time >= self.p.min_dwell_for_stop_sec
            and speed_px_sec < self.p.stop_disp_threshold_px_sec
        ):
            motion_state = "STOP"
        elif dwell_time >= 0.5 and speed_px_sec < self.p.slow_disp_threshold_px_sec:
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
    p: PKJIParams,
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
        (occupancy_pct >= p.occupancy_macet_pct and slow_or_stop_ratio >= 0.25)
        or (
            enough_for_macet
            and stop_count >= min_stopped_macet
            and stop_ratio >= p.stop_ratio_macet
        )
        or (
            occupancy_pct >= p.occupancy_padat_pct
            and stop_count >= min_stopped_macet
            and stop_ratio >= 0.35
        )
        or (
            enough_for_macet
            and dj_effective >= p.dj_macet_threshold
            and slow_or_stop_ratio >= 0.40
        )
    ):
        return "Macet (Congestion)"

    if (
        occupancy_pct >= p.occupancy_padat_pct
        or (
            enough_for_padat
            and (
                (stop_count >= min_stopped_padat and stop_ratio >= p.stop_ratio_padat)
                or slow_or_stop_ratio >= 0.50
                or (dj_effective >= p.dj_padat_threshold and occupancy_pct >= 20.0)
            )
        )
    ):
        return "Padat (Slow Traffic)"

    return "Lancar (Free Flow)"
