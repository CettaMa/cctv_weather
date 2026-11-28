"""Akumulator metrik per menit dan per video."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from .common import new_counts
from .config import CLASS_KEYS, PKJIParams


class MetricsAccumulator:
    """Mengakumulasi metrik frame demi frame untuk 1 bucket menit atau 1 video penuh."""

    def __init__(self, inference_interval: float, params: PKJIParams) -> None:
        self.p = params
        self.inference_interval = inference_interval
        # frame yang jendela arusnya sudah cukup panjang untuk dirata-ratakan
        self.flow_frames = 0
        self.sums = new_counts()
        self.motion_sums = {"Moving": 0, "SLOW": 0, "STOP": 0}
        self.status_counter: Counter = Counter()
        self.emp_mode_counter: Counter = Counter()

        self.sampled_frames = 0
        self.max_total = 0
        self.inference_time_total_ms = 0.0

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
        flow_valid: bool = True,
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
        if flow_valid:
            self.flow_frames += 1
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

    def summarize(self, processing_fps: float = 0.0) -> Dict[str, object]:
        n = self.sampled_frames
        nf = self.flow_frames

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
            est_flow_veh = round(self.flow_veh_sum / nf, 1) if nf else ""
            est_q_smp = round(self.q_smp_sum / nf, 1) if nf else ""
            avg_dj_fl = round(self.dj_flow_sum / nf, 3) if nf else ""
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
            proc_fps = round(processing_fps, 3)
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
            "c_roi_capacity_smp": round(self.p.roi_max_smp_capacity, 2),
            "avg_roi_occupancy_pct": avg_occ,
            "avg_pixel_occupancy_pct": avg_px_occ,
            "est_flow_veh_per_hour": est_flow_veh,
            "est_q_smp_per_hour": est_q_smp,
            "road_capacity_c_smp_per_hour": round(self.p.road_capacity_smp_per_hour, 1),
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
