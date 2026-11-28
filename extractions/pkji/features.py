"""Menghitung fitur PKJI dari cache deteksi (tanpa GPU).

Mode legacy: tiap video diproses sendiri, persis alur script lama.
Mode default: video satu lokasi diproses berurutan; state tracker/arus bersambung
bila videonya bersambung, bagian yang tumpang tindih dibuang, dan satu menit
kalender menghasilkan tepat satu baris.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .accumulate import MetricsAccumulator
from .cache import VideoDetections
from .common import map_vehicle_class, new_counts, summarize_error_messages, timestamp_local
from .config import PKJIParams
from .metrics import PKJIFlowAndOccupancyCalculator, VehicleMotionAnalyzer, classify_traffic_condition
from .tracking import LineCounter, MotionTracker, SimpleCentroidTracker

# video dianggap utuh bila frame terbaca >= rasio ini dari frame count container
COMPLETE_FRAME_RATIO = 0.98


def video_status(det: VideoDetections, sampled: int, legacy: bool) -> str:
    if sampled == 0:
        return "FAILED"
    if legacy:
        return "OK"
    return "OK" if is_complete(det) else "PARTIAL"


def is_complete(det: VideoDetections) -> bool:
    if det.failed:
        return False
    return det.frame_count <= 0 or det.frames_read >= det.frame_count * COMPLETE_FRAME_RATIO


class _Session:
    """State yang hidup selama pengamatan bersambung."""

    def __init__(
        self,
        legacy: bool,
        params: PKJIParams,
        interval: float,
        count_line: Optional[np.ndarray],
        start: float,
    ) -> None:
        self.legacy = legacy
        self.p = params
        self.start = start
        self.analyzer = VehicleMotionAnalyzer(params)
        self.calc = PKJIFlowAndOccupancyCalculator(params)
        # script lama membatasi interval minimal 0.05 s, yang memendekkan jendela di 0.04 s
        floor = 0.05 if legacy else 0.001
        smooth_len = max(1, int(round(params.traffic_state_smooth_sec / max(floor, interval))))
        self.recent_states: deque = deque(maxlen=smooth_len)
        self.last_cleanup_sec = None
        self.line: Optional[LineCounter] = None
        if legacy:
            self.tracker = SimpleCentroidTracker()
        else:
            self.tracker = MotionTracker(
                params.track_max_age_sec, params.track_gate, params.track_new_gate
            )
            if count_line is not None:
                self.line = LineCounter(count_line)

    def step(
        self,
        t: float,
        dets: np.ndarray,
        groups: Dict[int, Optional[str]],
        polygon: np.ndarray,
        roi_area: float,
    ) -> dict:
        """Memproses satu sampel pada waktu t (detik; legacy: sejak awal video)."""
        rel = t - self.start  # waktu sejak awal sesi pengamatan
        sec_int = int(t)
        if sec_int != self.last_cleanup_sec:
            self.analyzer.cleanup_stale(t)
            self.last_cleanup_sec = sec_int

        rows = dets.tolist()
        vehicles = [i for i, r in enumerate(rows) if groups.get(int(r[5])) is not None]
        inside = []
        for i in vehicles:
            x1, y1, x2, y2 = rows[i][:4]
            anchor = (float(int(round((x1 + x2) / 2.0))), float(int(round(y2))))
            if cv2.pointPolygonTest(polygon, anchor, False) >= 0:
                inside.append(i)

        if self.legacy:
            centroids = [
                ((rows[i][0] + rows[i][2]) / 2.0, (rows[i][1] + rows[i][3]) / 2.0) for i in inside
            ]
            ids = dict(zip(inside, self.tracker.update(centroids)))
        else:
            # lacak semua kendaraan di frame supaya ID tidak putus di tepi ROI
            all_ids = self.tracker.update(dets[vehicles], t)
            ids = dict(zip(vehicles, all_ids))
            if self.line is not None:
                for i in vehicles:
                    x1, _, x2, y2 = rows[i][:4]
                    if self.line.update(ids[i], ((x1 + x2) / 2.0, y2)):
                        self.calc.register_departure(rel, ids[i], groups[int(rows[i][5])])

        frame_counts = new_counts()
        motion_counts = {"Moving": 0, "SLOW": 0, "STOP": 0}
        active: List[Tuple[int, str, float]] = []
        bbox_area = 0.0
        for i in inside:
            x1, y1, x2, y2 = rows[i][:4]
            group = groups[int(rows[i][5])]
            frame_counts[group] += 1
            bbox_area += max(0.0, (x2 - x1) * (y2 - y1))
            track_id = ids[i]
            info = self.analyzer.update_vehicle(
                int(track_id), t, int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
            )
            motion_counts[str(info["motion_state"])] += 1
            active.append((int(track_id), group, float(info["dwell_time"])))

        if self.line is None:
            if not self.legacy and rel <= self.p.initial_track_grace_sec:
                for track_id, _, _ in active:
                    self.calc.exclude_track(track_id)
            self.calc.register_active_tracks(rel, active)
        flow_valid = self.legacy or rel >= self.p.min_flow_extrapolation_window_sec
        metrics = self.calc.compute_pkji_metrics(rel, frame_counts)
        if not flow_valid:  # arus belum bisa diestimasi: DJ memakai okupansi ruang
            metrics["dj_effective"] = metrics["dj_spatial"]

        occupancy = float(metrics["roi_occupancy_pct"])
        instant = classify_traffic_condition(
            occupancy,
            float(metrics["dj_effective"]),
            sum(frame_counts.values()),
            motion_counts["Moving"],
            motion_counts["SLOW"],
            motion_counts["STOP"],
            self.p,
        )
        self.recent_states.append(instant)
        return {
            "frame_counts": frame_counts,
            "motion_counts": motion_counts,
            "pkji_metrics": metrics,
            "pixel_occupancy_pct": min(100.0, (bbox_area / roi_area) * 100.0),
            "smoothed_status": Counter(self.recent_states).most_common(1)[0][0],
            "flow_valid": flow_valid,
        }


class _MinuteBucket:
    def __init__(self, bucket: datetime, interval: float, params: PKJIParams) -> None:
        self.bucket = bucket
        self.acc = MetricsAccumulator(interval, params)
        self.videos: List[str] = []
        self.incomplete = False
        self.fps_weighted = 0.0

    def add(self, det: VideoDetections, video_sec: float, frame_dt: datetime, out: dict, ms: float) -> None:
        if det.video_name not in self.videos:
            self.videos.append(det.video_name)
            for err in det.errors:
                self.acc.add_error(err)
        self.incomplete |= not is_complete(det)
        self.fps_weighted += det.processing_fps
        self.acc.add_frame(video_sec=video_sec, frame_dt=frame_dt, inference_ms=ms, **out)

    def row(self, location: str, legacy: bool) -> dict:
        n = self.acc.sampled_frames
        stats = self.acc.summarize(self.fps_weighted / n if n else 0.0)
        dominant = stats.pop("dominant_cond")
        duration = self.acc.video_span_seconds()
        if n == 0:
            status = "FAILED"
        elif duration < 45.0 or n < 30 or (self.incomplete and not legacy):
            status = "PARTIAL"
        else:
            status = "OK"
        return {
            "video_file": "|".join(self.videos),
            "location": location,
            "hour_bucket": self.bucket.replace(minute=0).isoformat(timespec="seconds"),
            "one_min_bucket": self.bucket.isoformat(timespec="seconds"),
            "session_point": self.bucket.minute + 1,
            "sample_start_local": timestamp_local(self.acc.first_dt),
            "sample_end_local": timestamp_local(self.acc.last_dt),
            "target_sample_duration_seconds": 60,
            "actual_sample_duration_seconds": duration,
            **stats,
            "dominant_traffic_condition": dominant,
            "one_min_traffic_condition": dominant,
            "status": status,
            "error_message": summarize_error_messages(self.acc.errors),
            "_flow_frames": self.acc.flow_frames,  # bobot agregasi; tidak ditulis ke CSV
        }


class FeatureExtractor:
    """Memproses deteksi video-video satu lokasi, dalam urutan waktu mulai."""

    def __init__(
        self,
        location: str,
        params: PKJIParams,
        roi_loader,
        interval: float,
        legacy: bool,
    ) -> None:
        self.location = location
        self.p = params
        self.roi_loader = roi_loader  # (width, height) -> (polygon, count_line)
        self.interval = interval
        self.legacy = legacy
        self.minute_rows: List[dict] = []
        self.video_rows: List[dict] = []
        self._session: Optional[_Session] = None
        self._bucket: Optional[_MinuteBucket] = None
        self._last_abs: Optional[float] = None
        self._epoch: Optional[datetime] = None

    def snapshot(self) -> dict:
        """State lengkap untuk checkpoint (tanpa roi_loader yang tidak bisa di-pickle)."""
        return {k: v for k, v in self.__dict__.items() if k != "roi_loader"}

    def restore(self, state: dict) -> None:
        self.__dict__.update(state)

    def _flush_bucket(self) -> None:
        if self._bucket is not None and self._bucket.acc.sampled_frames:
            self.minute_rows.append(self._bucket.row(self.location, self.legacy))
        self._bucket = None

    def add_video(self, det: VideoDetections, start_dt: datetime, check=lambda: None) -> None:
        """`check()` dipanggil berkala; lempar exception di sana untuk menghentikan."""
        video_acc = MetricsAccumulator(self.interval, self.p)
        for err in det.errors:
            video_acc.add_error(err)
        groups = {cid: map_vehicle_class(name) for cid, name in det.names.items()}
        polygon = line = None
        roi_area = 1.0
        if len(det.frame_idx):
            polygon, line = self.roi_loader(det.width, det.height)
            roi_area = max(1.0, float(cv2.contourArea(polygon)))
        if self._epoch is None:
            self._epoch = start_dt
        if self.legacy:
            self._flush_bucket()
            self._session = None
        skipped = 0

        for k in range(len(det.frame_idx)):
            if k % 256 == 0:
                check()
            if self.legacy:
                video_sec = int(det.frame_idx[k]) / det.fps
                t = video_sec
            else:
                video_sec = float(det.pos_sec[k])
                t = (start_dt - self._epoch).total_seconds() + video_sec
                if self._last_abs is not None and t <= self._last_abs + 1e-6:
                    skipped += 1  # sudah tercakup video sebelumnya
                    continue
                if self._last_abs is None or t - self._last_abs > self.p.session_max_gap_sec:
                    self._session = None
                self._last_abs = t
            if self._session is None:
                self._session = _Session(
                    self.legacy, self.p, self.interval, line, 0.0 if self.legacy else t
                )
            frame_dt = start_dt + timedelta(seconds=video_sec)
            out = self._session.step(t, det.detections(k), groups, polygon, roi_area)
            ms = float(det.infer_ms[k])
            video_acc.add_frame(video_sec=video_sec, frame_dt=frame_dt, inference_ms=ms, **out)

            bucket = frame_dt.replace(second=0, microsecond=0)
            if self._bucket is None or self._bucket.bucket != bucket:
                self._flush_bucket()
                self._bucket = _MinuteBucket(bucket, self.interval, self.p)
            minute_t = video_sec if self.legacy else t
            self._bucket.add(det, minute_t, frame_dt, out, ms)

        if self.legacy:
            self._flush_bucket()
            duration = round(det.frames_read / det.fps, 3) if det.frames_read else 0.0
        else:
            if skipped:
                video_acc.add_error(f"{skipped} sampel tumpang tindih dengan video sebelumnya dilewati")
            duration = round(float(det.end_sec), 3)
        stats = video_acc.summarize(det.processing_fps)
        condition = stats.pop("dominant_cond")
        self.video_rows.append(
            {
                "video_file": det.video_name,
                "location": self.location,
                "video_start_local": timestamp_local(start_dt),
                "video_end_local": timestamp_local(start_dt + timedelta(seconds=duration)),
                "video_duration_seconds": duration,
                "video_fps": round(det.fps, 3),
                **stats,
                "video_traffic_condition": condition,
                "status": video_status(det, video_acc.sampled_frames, self.legacy),
                "error_message": summarize_error_messages(video_acc.errors),
            }
        )

    def finish(self) -> None:
        self._flush_bucket()
