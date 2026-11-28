"""Agregasi baris 1 menit menjadi interval 5-60 menit."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from .common import append_csv_row, open_csv_writer, summarize_error_messages
from .config import ALL_AGGREGATION_INTERVALS, CLASS_KEYS, PKJIParams
from .metrics import classify_traffic_condition
from .schemas import HOURLY_CSV_FIELDS, INTERVAL_CSV_FIELDS

FLOW_COLUMNS = {"est_flow_veh_per_hour", "est_q_smp_per_hour", "avg_dj_flow"}


def compute_aggregate_metrics(
    sessions: List[dict], expected_sessions: int, params: PKJIParams, weighted: bool
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

        def mean_metric(col: str, ndigits: int = 3):
            if not weighted:  # --legacy: tiap baris menit berbobot sama
                return round(
                    sum(float(s[col]) for s in valid_sessions) / n_valid, ndigits
                )
            weight_key = "_flow_frames" if col in FLOW_COLUMNS else "sampled_frames"
            pairs = [
                (float(s[col]), int(s[weight_key]))
                for s in valid_sessions
                if s[col] != ""
            ]
            total_weight = sum(w for _, w in pairs)
            if not total_weight:
                return ""
            return round(sum(v * w for v, w in pairs) / total_weight, ndigits)

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
            est_flow_veh != ""
            and est_flow_veh >= params.emp_switch_flow_veh_per_hour
        ) or avg_occ >= params.initial_occupancy_threshold_for_high_emp_pct:
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
            avg_occ, avg_dj_ef, avg_total, avg_moving, avg_slow, avg_stop, params
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
        "c_roi_capacity_smp": round(params.roi_max_smp_capacity, 2),
        "avg_roi_occupancy_pct": avg_occ,
        "avg_pixel_occupancy_pct": avg_px_occ,
        "est_flow_veh_per_hour": est_flow_veh,
        "est_q_smp_per_hour": est_q_smp,
        "road_capacity_c_smp_per_hour": round(params.road_capacity_smp_per_hour, 1),
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
    params: PKJIParams,
    weighted: bool,
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
                metrics = compute_aggregate_metrics(sessions, expected_n, params, weighted)
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
