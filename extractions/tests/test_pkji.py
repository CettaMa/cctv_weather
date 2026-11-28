import threading
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from pkji import cache
from pkji.checkpoint import FeatureCheckpoint
from pkji.aggregate import compute_aggregate_metrics
from pkji.cli import check_append, parse_args
from pkji.config import PKJIParams, TIMEZONE, load_roi
from pkji.detector import letterbox_geometry
from pkji.features import FeatureExtractor
from pkji.metrics import PKJIFlowAndOccupancyCalculator
from pkji.pipeline import Job, detect_video
from pkji.tracking import LineCounter, MotionTracker

NAMES = {0: "motor", 1: "mobil", 2: "truk", 3: "orang", 4: "bis"}
ROI = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]], np.int32)
START = datetime(2026, 11, 11, 11, 0, 0, tzinfo=TIMEZONE)


def make_video(name, seconds, boxes_at, **meta):
    """Video sintetis 5 sampel/detik; boxes_at(t) -> list [x1,y1,x2,y2,conf,cls]."""
    n = int(seconds * 5)
    per_sample = [np.array(boxes_at(k * 0.2), np.float32).reshape(-1, 6) for k in range(n)]
    values = dict(fps=25.0, frame_count=n * 5, frames_read=n * 5, width=640, height=480,
                  end_sec=seconds, processing_fps=100.0)
    values.update(meta)
    return cache.VideoDetections(
        video_name=name, names=NAMES,
        frame_idx=np.arange(n, dtype=np.int32) * 5,
        pos_sec=np.arange(n) * 0.2,
        infer_ms=np.ones(n, np.float32),
        boxes=np.concatenate(per_sample) if n else np.zeros((0, 6), np.float32),
        offsets=np.concatenate(([0], np.cumsum([len(b) for b in per_sample]))).astype(np.int32),
        **values,
    )


def parked_car(_t):
    return [[100, 100, 160, 140, 0.9, 1]]


def extractor(legacy, line=None, params=None):
    return FeatureExtractor("LOKASI", params or PKJIParams(), lambda w, h: (ROI, line),
                            0.2, legacy)


def run(legacy, videos, **kwargs):
    ex = extractor(legacy, **kwargs)
    for det, start in videos:
        ex.add_video(det, start)
    ex.finish()
    return ex


def test_default_params_match_old_constants():
    p = PKJIParams()
    assert p.road_capacity_smp_per_hour == pytest.approx(3264.0)
    assert p.roi_max_smp_capacity == pytest.approx(24.0)
    assert p.emp_switch_flow_veh_per_hour == 2100.0
    with pytest.raises(ValueError):
        PKJIParams.from_dict({"salah": 1})


def test_flow_extrapolates_per_hour():
    calc = PKJIFlowAndOccupancyCalculator(PKJIParams())
    calc.register_active_tracks(40.0, [(1, "mobil", 1.0), (2, "motor", 0.1)])
    metrics = calc.compute_pkji_metrics(40.0, {"motor": 0, "mobil": 1, "bus": 0, "truk": 0})
    assert metrics["flow_veh_per_hour"] == pytest.approx(90.0)


def test_initial_vehicle_counted_only_in_legacy():
    video = make_video("a.mp4", 50, parked_car)
    legacy = run(True, [(video, START)]).video_rows[0]
    default = run(False, [(video, START)]).video_rows[0]
    assert legacy["est_flow_veh_per_hour"] > 0
    assert default["est_flow_veh_per_hour"] == 0
    assert default["average_active_mobil"] == 1.0


def test_default_merges_shared_minute_and_drops_overlap():
    first = make_video("a.mp4", 30, parked_car)
    second = make_video("b.mp4", 30, parked_car)
    videos = [(first, START), (second, START + timedelta(seconds=28))]
    legacy = run(True, videos).minute_rows
    default = run(False, videos).minute_rows
    assert len(legacy) == 2
    assert len(default) == 1
    assert default[0]["video_file"] == "a.mp4|b.mp4"
    assert default[0]["sampled_frames"] == 150 + 150 - 10


def test_state_carries_over_contiguous_videos():
    videos = [(make_video("a.mp4", 60, parked_car), START),
              (make_video("b.mp4", 60, parked_car), START + timedelta(seconds=60))]
    rows = run(False, videos).video_rows
    assert rows[1]["est_flow_veh_per_hour"] != ""  # tidak perlu pemanasan ulang
    assert rows[1]["avg_stopped_vehicles"] == pytest.approx(1.0)


def test_gap_starts_new_session():
    videos = [(make_video("a.mp4", 20, parked_car), START),
              (make_video("b.mp4", 20, parked_car), START + timedelta(hours=1))]
    rows = run(False, videos).video_rows
    assert rows[1]["est_flow_veh_per_hour"] == ""  # sesi baru, arus belum valid


def test_truncated_video_status():
    video = make_video("a.mp4", 50, parked_car, frame_count=10000)
    assert run(True, [(video, START)]).video_rows[0]["status"] == "OK"
    default = run(False, [(video, START)])
    assert default.video_rows[0]["status"] == "PARTIAL"
    assert default.minute_rows[0]["status"] == "PARTIAL"


def test_line_counter_counts_once():
    counter = LineCounter(np.array([[0, 100], [200, 100]], np.float32))
    positions = [(50, 80), (50, 100), (50, 120), (50, 90)]
    assert sum(counter.update(7, p) for p in positions) == 1


def test_line_crossing_flow():
    def moving(t):
        y = 100 + t * 5
        return [[100, y - 40, 160, y, 0.9, 1]]

    line = np.array([[0, 300], [639, 300]], np.float32)
    row = run(False, [(make_video("a.mp4", 50, moving), START)], line=line).video_rows[0]
    assert row["est_flow_veh_per_hour"] > 0


def session(minute, frames, total, status="OK"):
    row = {k: 0 for k in (
        "average_active_motor", "average_active_mobil", "average_active_bus", "average_active_truk",
        "avg_smp_frame", "avg_roi_occupancy_pct", "avg_pixel_occupancy_pct", "est_flow_veh_per_hour",
        "est_q_smp_per_hour", "avg_dj_flow", "avg_dj_spatial", "avg_dj_effective", "max_dj_effective",
        "avg_moving_vehicles", "avg_slow_vehicles", "avg_stopped_vehicles", "pct_time_lancar",
        "pct_time_padat", "pct_time_macet", "processing_fps")}
    row.update(one_min_bucket=minute, sampled_frames=frames, average_active_total=total,
               max_active_total=total, average_inference_ms=1.0, target_sample_duration_seconds=60,
               actual_sample_duration_seconds=frames / 5, status=status, error_message="",
               _flow_frames=frames)
    return row


def test_weighted_aggregation():
    rows = [session("a", 300, 10), session("b", 10, 40, "PARTIAL")]
    assert compute_aggregate_metrics(rows, 2, PKJIParams(), False)["average_active_total"] == 25
    weighted = compute_aggregate_metrics(rows, 2, PKJIParams(), True)["average_active_total"]
    assert weighted == pytest.approx((300 * 10 + 10 * 40) / 310, abs=1e-3)


def test_aggregation_ignores_rows_without_flow():
    cold = session("a", 100, 5)
    cold.update(est_flow_veh_per_hour="", est_q_smp_per_hour="", avg_dj_flow="", _flow_frames=0)
    warm = session("b", 300, 5)
    warm["est_flow_veh_per_hour"] = 600
    result = compute_aggregate_metrics([cold, warm], 2, PKJIParams(), True)
    assert result["est_flow_veh_per_hour"] == 600


def test_cache_roundtrip(tmp_path):
    video = make_video("a.mp4", 3, lambda t: parked_car(t) if t > 1 else [])
    cache.save(tmp_path / "a.parquet", video)
    loaded = cache.load(tmp_path / "a.parquet")
    assert loaded.names == NAMES
    np.testing.assert_array_equal(loaded.offsets, video.offsets)
    np.testing.assert_array_equal(loaded.boxes, video.boxes)
    assert cache.is_reusable(tmp_path / "a.parquet")
    video.failed = True
    cache.save(tmp_path / "a.parquet", video)
    assert not cache.is_reusable(tmp_path / "a.parquet")


def test_append_rejects_other_method(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"method": "legacy"}')
    check_append(tmp_path, {"method": "legacy"})
    with pytest.raises(RuntimeError):
        check_append(tmp_path, {"method": "default"})


def test_letterbox_geometry():
    assert letterbox_geometry(480, 640, 640, rect=False)[3] == (0, 0, 80, 80)
    assert letterbox_geometry(480, 640, 640, rect=True)[3] == (0, 0, 0, 0)


def test_roi_count_line_scaled(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text('{"frame_width":640,"frame_height":480,'
                    '"points":[[0,0],[600,0],[600,400]],"count_line":[[0,200],[320,200]]}')
    _, line = load_roi(path, 1280, 960)
    np.testing.assert_allclose(line, [[0, 400], [640, 400]])


class _FakeClient:
    """Satu 'deteksi' per frame, posisinya mengikuti isi frame; bisa minta stop."""

    names = NAMES

    def __init__(self, stop=None, stop_after=0):
        self.stop, self.stop_after, self.seen = stop, stop_after, 0

    def submit(self, frames):
        self.seen += len(frames)
        if self.stop is not None and self.seen >= self.stop_after:
            self.stop.set()
        return [np.array([[f.mean(), 0, 1, 1, 0.9, 1]], np.float32) for f in frames], 1.0

    def fetch(self, ticket):
        return ticket


def test_detection_resumes_mid_video(tmp_path):
    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(500):
        writer.write(np.full((48, 64, 3), i % 250, np.uint8))
    writer.release()

    full = detect_video(Job(str(video), str(tmp_path / "full.parquet")), _FakeClient(), 0.2, 8,
                        threading.Event())
    job = Job(str(video), str(tmp_path / "part.parquet"))
    partial = cache.partial_path(Path(job.cache_file))
    for stop_after in (24, 40):  # dua kali pause
        stop = threading.Event()
        assert detect_video(job, _FakeClient(stop, stop_after), 0.2, 8, stop) is None
        assert partial.is_file()
    assert 0 < len(cache.load(partial).frame_idx) < len(full.frame_idx)
    resumed = detect_video(job, _FakeClient(), 0.2, 8, threading.Event())
    for name in ("frame_idx", "pos_sec", "offsets", "boxes"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(full, name))
    assert resumed.frames_read == full.frames_read == 500
    cache.save_final(Path(job.cache_file), resumed)
    assert not partial.exists()


@pytest.mark.parametrize("legacy", [True, False])
def test_feature_checkpoint_matches_uninterrupted(tmp_path, legacy):
    videos = [(make_video("a.mp4", 50, parked_car), START),
              (make_video("b.mp4", 50, parked_car), START + timedelta(seconds=50))]
    expected = run(legacy, videos)

    cache_file = tmp_path / "a.parquet"
    cache.save(cache_file, videos[0][0])
    store = FeatureCheckpoint(tmp_path, "LOKASI", {"method": legacy}, b"roi")
    first = extractor(legacy)
    first.add_video(*videos[0])
    store.save([store.entry(cache_file)], first.snapshot())

    done, state = store.load([Path("a")], {Path("a"): cache_file}, lambda v: None)
    assert len(done) == 1
    second = extractor(legacy)
    second.restore(state)
    second.add_video(*videos[1])
    second.finish()
    assert second.video_rows == expected.video_rows
    assert second.minute_rows == expected.minute_rows

    videos[0][0].processing_fps += 1  # cache berubah -> checkpoint tidak berlaku
    cache.save(cache_file, videos[0][0])
    assert store.load([Path("a")], {Path("a"): cache_file}, lambda v: None) == ([], None)


def test_default_interval_depends_on_method():
    assert parse_args(["-i", "x"]).inference_interval == 0.04
    assert parse_args(["-i", "x", "--legacy"]).inference_interval == 0.2
    assert parse_args(["-i", "x", "--legacy", "--inference-interval", "0.1"]).inference_interval == 0.1


def boxes(*lefts, top=100, size=40):
    return np.array([[x, top, x + size, top + size, 0.9, 0] for x in lefts], np.float32).reshape(-1, 6)


def test_tracker_follows_fast_vehicle_without_overlap():
    tracker = MotionTracker()
    # kecepatan membesar sampai satu lebar kotak per sampel (kotak tidak beririsan)
    positions = [0, 30, 65, 105, 145, 185, 225]
    ids = [tracker.update(boxes(x), k * 0.2)[0] for k, x in enumerate(positions)]
    assert len(set(ids)) == 1


def test_tracker_keeps_neighbours_apart():
    tracker = MotionTracker()
    first = tracker.update(boxes(0, 100), 0.0)
    for k in range(1, 8):
        # urutan deteksi dibalik tiap sampel ganjil
        lefts = [k * 20, 100 + k * 20]
        ids = tracker.update(boxes(*(lefts[::-1] if k % 2 else lefts)), k * 0.2)
        assert (ids[::-1] if k % 2 else ids) == first


def test_tracker_survives_short_gap_and_expires():
    tracker = MotionTracker(max_age_sec=0.8)
    car = tracker.update(boxes(100), 0.0)[0]
    tracker.update(boxes(), 0.2)
    assert tracker.update(boxes(100), 0.6)[0] == car
    assert tracker.update(boxes(100), 2.0)[0] != car


def test_tracker_does_not_jump_to_distant_detection():
    tracker = MotionTracker()
    car = tracker.update(boxes(0), 0.0)[0]
    assert tracker.update(boxes(300), 0.2)[0] != car
