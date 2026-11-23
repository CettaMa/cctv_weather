#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record fixed-length clips from Pelindung Bandung CCTV streams using ffmpeg
in copy mode (no re-encode), then verify the resulting file duration.

Usage:
    python script_tools/record_cctv_clips.py
    python script_tools/record_cctv_clips.py --targets "SP BUAHBATU" --duration 30
    python script_tools/record_cctv_clips.py --targets "SP BUAHBATU" --duration 30 \
        --max-part-seconds 15 --keep-parts
    python script_tools/record_cctv_clips.py --targets "SP BUAHBATU" --duration 0
        (record until Ctrl+C / Ctrl+Break; recorded footage is muxed and verified)

Exit codes:
    0 - every requested camera produced a PASS clip
    1 - fatal error (unresolved name, missing ffmpeg/ffprobe, or zero PASS)
    2 - partial success (at least one PASS, but some camera is DEAD/FAIL)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_bandung_cctv import (  # noqa: E402
    API_URL,
    normalize_stream_url,
    request_headers,
    sanitize_filename,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "cctv_clips"
FALLBACK_CSV = BASE_DIR / "csv" / "cctv_bandung_data_validated.csv"
DEFAULT_TARGETS = ("SP BUAHBATU", "CCTV SOETA-BUBAT 01", "CCTV SOETA-BUBAT 03")
DEFAULT_DURATION = 600
DEFAULT_TOLERANCE = 1.5
STALE_PLAYLIST_SECONDS = 3600
STOP_EVENT = threading.Event()
FFMPEG_POPEN_KWARGS = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32"
    else {"start_new_session": True}
)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("\u00a0", " ").strip()).upper()


@dataclass(frozen=True)
class Camera:
    name: str
    stream_url: str
    provider: str


@dataclass
class Preflight:
    playlist_status: object = ""
    playlist_last_modified: str = ""
    playlist_age_seconds: float | None = None
    has_endlist: bool = False
    segment_count: int = 0
    first_segment_url: str = ""
    first_segment_status: object = ""
    alive: bool = False
    reason: str = ""


@dataclass
class RecordResult:
    status: str
    output_file: str = ""
    duration_seconds: float | None = None
    size_bytes: int = 0
    codec_name: str = ""
    width: int = 0
    height: int = 0
    avg_frame_rate: str = ""
    nb_read_packets: int = 0
    parts: int = 0
    wall_seconds: float = 0.0
    error: str = ""


def http_get(url: str, timeout: float, headers: dict[str, str] | None = None, range_bytes: str | None = None):
    req_headers = dict(request_headers())
    if headers:
        req_headers.update(headers)
    if range_bytes:
        req_headers["Range"] = range_bytes
    request = Request(url, headers=req_headers, method="GET")
    return urlopen(request, timeout=timeout)


def resolve_targets(target_names: list[str]) -> list[Camera] | None:
    wanted = {normalize_name(n): n for n in target_names}
    resolved: dict[str, Camera] = {}
    catalogue: list[tuple[str, str, str]] = []  # (normalized name, provider, stream_url)
    source = "api"

    try:
        with http_get(API_URL, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list, got {type(payload).__name__}")
        for item in payload:
            raw_name = str(item.get("cctv_name") or item.get("name") or "")
            norm = normalize_name(raw_name)
            if not norm:
                continue
            stream_url = normalize_stream_url(item.get("stream_cctv") or item.get("stream_url"))
            provider = str(item.get("dinas") or "").strip() or "UNKNOWN"
            catalogue.append((norm, raw_name.strip(), provider))
            if norm in wanted and norm not in resolved:
                if not stream_url:
                    logging.warning("%s: API returned empty stream URL", raw_name.strip())
                resolved[norm] = Camera(name=wanted[norm], stream_url=stream_url, provider=provider)
    except (URLError, HTTPError, TimeoutError, ValueError, OSError) as exc:
        logging.warning("Live API unreachable (%s); falling back to %s", exc, FALLBACK_CSV)
        source = "csv"
        if not FALLBACK_CSV.exists():
            logging.error("Fallback CSV not found: %s", FALLBACK_CSV)
            return None
        with FALLBACK_CSV.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_name = str(row.get("CCTV_Name") or "")
                norm = normalize_name(raw_name)
                if not norm:
                    continue
                stream_url = normalize_stream_url(row.get("m3u8_URL"))
                provider = str(row.get("Provider") or "").strip() or "UNKNOWN"
                catalogue.append((norm, raw_name.strip(), provider))
                if norm in wanted and norm not in resolved:
                    resolved[norm] = Camera(name=wanted[norm], stream_url=stream_url, provider=provider)

    missing = [orig for norm, orig in wanted.items() if norm not in resolved]
    if missing:
        for orig in missing:
            tokens = normalize_name(orig).split()
            candidates = sorted(
                {orig_name for norm, orig_name, _ in catalogue if any(t in norm for t in tokens)}
            )[:10]
            logging.error(
                "Could not resolve camera %r. Closest candidates: %s",
                orig,
                candidates or "(none found)",
            )
        return None

    logging.info("Resolved %d target(s) from %s", len(resolved), source)
    return [resolved[normalize_name(n)] for n in target_names]


def preflight(stream_url: str) -> Preflight:
    pf = Preflight()
    if not stream_url:
        pf.reason = "no stream URL"
        return pf

    try:
        with http_get(stream_url, timeout=15) as response:
            pf.playlist_status = response.status
            last_modified = response.headers.get("Last-Modified", "")
            pf.playlist_last_modified = last_modified
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        pf.playlist_status = exc.code
        pf.reason = f"playlist HTTP {exc.code}"
        return pf
    except (URLError, TimeoutError, OSError) as exc:
        pf.playlist_status = "error"
        pf.reason = f"playlist fetch failed: {exc}"
        return pf

    if pf.playlist_status != 200:
        pf.reason = f"playlist HTTP {pf.playlist_status}"
        return pf

    if pf.playlist_last_modified:
        try:
            lm_dt = parsedate_to_datetime(pf.playlist_last_modified)
            if lm_dt.tzinfo is None:
                lm_dt = lm_dt.replace(tzinfo=timezone.utc)
            pf.playlist_age_seconds = (datetime.now(timezone.utc) - lm_dt).total_seconds()
        except (TypeError, ValueError):
            pf.playlist_age_seconds = None

    pf.has_endlist = "#EXT-X-ENDLIST" in body
    segments = [line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")]
    pf.segment_count = len(segments)

    if pf.segment_count == 0:
        pf.reason = "playlist has no segments"
        return pf

    pf.first_segment_url = urljoin(stream_url, segments[0])
    try:
        with http_get(pf.first_segment_url, timeout=15, range_bytes="bytes=0-1023") as response:
            pf.first_segment_status = response.status
    except HTTPError as exc:
        pf.first_segment_status = exc.code
    except (URLError, TimeoutError, OSError) as exc:
        pf.first_segment_status = f"error: {exc}"

    if pf.first_segment_status not in (200, 206):
        pf.reason = f"first segment HTTP {pf.first_segment_status}"
        return pf

    if pf.playlist_age_seconds is not None and pf.playlist_age_seconds > STALE_PLAYLIST_SECONDS:
        hours = pf.playlist_age_seconds / 3600
        pf.reason = f"playlist stale {hours:.1f}h (Last-Modified {pf.playlist_last_modified})"
        return pf

    pf.alive = True
    return pf


def probe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0.0
    value = (result.stdout or "").strip()
    try:
        duration = float(value)
    except ValueError:
        return 0.0
    return duration if duration > 0.05 else 0.0


def run_ffmpeg_part(cmd: list[str], timeout: float | None) -> tuple[int, str]:
    """Run ffmpeg isolated from console signals; on STOP_EVENT send 'q' so it finalizes the file."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, text=True, **FFMPEG_POPEN_KWARGS
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        rc: int | None = None
        while rc is None:
            if STOP_EVENT.wait(0.5):
                try:
                    proc.communicate(input="q", timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                rc = proc.returncode
                break
            rc = proc.poll()
            if rc is None and deadline is not None and time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                err.seek(0)
                return -1, f"ffmpeg timed out after {timeout:.0f}s; {err.read()}"
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        err.seek(0)
        return rc, err.read()


def record_clip(cam: Camera, target_seconds: float, out_path: Path, parts_dir: Path, max_part_seconds: float) -> RecordResult:
    """Record `target_seconds` of footage; target_seconds <= 0 records until STOP_EVENT is set."""
    stem = sanitize_filename(cam.name)
    parts_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    unbounded = target_seconds <= 0
    wall_budget = target_seconds * 2 + 600

    parts: list[Path] = []
    accumulated = 0.0
    consecutive_failures = 0
    part_index = 0
    result = RecordResult(status="FAIL")

    while (unbounded or accumulated < target_seconds - 0.5) and not STOP_EVENT.is_set():
        if not unbounded and time.monotonic() - started > wall_budget:
            result.error = f"wall-clock budget exceeded ({wall_budget:.0f}s) with {accumulated:.1f}s recorded"
            break
        if unbounded:
            part_seconds = max_part_seconds if max_part_seconds > 0 else None
        else:
            remaining = target_seconds - accumulated
            part_seconds = remaining if max_part_seconds <= 0 else min(remaining, max_part_seconds)
        part_path = parts_dir / f"{stem}_part{part_index:02d}.ts"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-user_agent", request_headers()["User-Agent"],
            "-headers", "Referer: https://pelindung.bandung.go.id/\r\n",
            "-rw_timeout", "15000000",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", cam.stream_url,
            *(["-t", str(part_seconds)] if part_seconds is not None else []),
            "-an", "-c", "copy", "-f", "mpegts", str(part_path),
        ]
        timeout = None if part_seconds is None else part_seconds * 2 + 180
        rc, stderr = run_ffmpeg_part(cmd, timeout)
        stopped = STOP_EVENT.is_set()

        part_duration = probe_duration(part_path) if (rc == 0 or stopped) and part_path.exists() else 0.0

        if stopped and part_duration > 0.0:
            accumulated += part_duration
            parts.append(part_path)
            break

        if rc != 0 or part_duration <= 0.0:
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            consecutive_failures += 1
            result.error = stderr[:500]
            if consecutive_failures >= 3 or stopped:
                break
            STOP_EVENT.wait(3)
            continue

        consecutive_failures = 0
        accumulated += part_duration
        parts.append(part_path)
        part_index += 1
        if unbounded:
            logging.info("%s: part %d done, %.1fs recorded so far", cam.name, part_index, accumulated)

    result.wall_seconds = time.monotonic() - started
    result.parts = len(parts)

    if not parts:
        result.status = "FAIL"
        if not result.error:
            result.error = "no parts recorded"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(parts) == 1:
        mux_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(parts[0]), "-an", "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]
    else:
        list_path = parts_dir / f"{stem}_concat.txt"
        with list_path.open("w", encoding="utf-8") as handle:
            for part in parts:
                handle.write(f"file '{part.resolve().as_posix()}'\n")
        mux_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-an", "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]

    try:
        mux_proc = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        result.status = "FAIL"
        result.error = f"mux timed out after {exc.timeout:.0f}s"
        return result

    if mux_proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        result.status = "FAIL"
        result.error = (mux_proc.stderr or "mux failed")[:500]
        return result

    result.status = "OK"
    result.output_file = str(out_path)
    result.duration_seconds = accumulated
    return result


def verify_clip(out_path: Path, target_seconds: float, tolerance: float) -> dict:
    """target_seconds <= 0 (record-until-stopped) passes any decodable clip with positive duration."""
    verdict: dict = {
        "status": "FAIL",
        "duration_seconds": None,
        "size_bytes": 0,
        "codec_name": "",
        "width": 0,
        "height": 0,
        "avg_frame_rate": "",
        "nb_read_packets": 0,
        "error": "",
    }
    if not out_path.exists() or out_path.stat().st_size == 0:
        verdict["error"] = "output file missing or empty"
        return verdict

    verdict["size_bytes"] = out_path.stat().st_size
    cmd = [
        "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
        "-show_entries",
        "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_read_packets",
        "-of", "json", str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(proc.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        verdict["error"] = f"ffprobe failed: {exc}"
        return verdict

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    stream = streams[0] if streams else {}

    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    verdict["duration_seconds"] = duration
    verdict["codec_name"] = stream.get("codec_name", "")
    verdict["width"] = stream.get("width", 0)
    verdict["height"] = stream.get("height", 0)
    verdict["avg_frame_rate"] = stream.get("avg_frame_rate", "")
    try:
        verdict["nb_read_packets"] = int(stream.get("nb_read_packets", 0))
    except (TypeError, ValueError):
        verdict["nb_read_packets"] = 0

    if target_seconds <= 0:
        if duration > 0 and verdict["nb_read_packets"] > 0:
            verdict["status"] = "PASS"
        else:
            verdict["error"] = "clip has no decodable video"
    elif abs(duration - target_seconds) <= tolerance:
        verdict["status"] = "PASS"
    else:
        verdict["error"] = f"duration {duration:.3f}s outside target {target_seconds:.1f}s +/-{tolerance}s"

    return verdict


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record CCTV clips from Pelindung Bandung in ffmpeg copy mode.")
    parser.add_argument("--targets", action="append", help="Camera name (repeatable). Default: the 3 standard targets.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Target clip duration in seconds; 0 = record until Ctrl+C.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Allowed duration deviation in seconds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for clips and reports.")
    parser.add_argument("--max-part-seconds", type=float, default=0, help="Force recording to split into parts of this length (0 = no limit).")
    parser.add_argument("--keep-parts", action="store_true", help="Keep .ts parts and concat list after muxing.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip liveness preflight; attempt to record every target.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args(argv if argv is not None else sys.argv[1:])
    target_names = args.targets or list(DEFAULT_TARGETS)
    output_dir = args.output_dir if args.output_dir.is_absolute() else BASE_DIR / args.output_dir
    parts_dir = output_dir / "parts"

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        logging.error("ffmpeg and ffprobe must both be on PATH")
        return 1

    cameras = resolve_targets(target_names)
    if cameras is None:
        return 1

    started_utc = datetime.now(timezone.utc)
    clips: list[dict] = []
    live_jobs: dict[str, tuple[Camera, Path]] = {}

    for cam in cameras:
        pf = Preflight(alive=True) if args.skip_preflight else preflight(cam.stream_url)
        clip_entry = {
            "cctv_name": cam.name,
            "provider": cam.provider,
            "stream_url": cam.stream_url,
            "status": "DEAD",
            "output_file": "",
            "duration_seconds": None,
            "size_bytes": 0,
            "codec_name": "",
            "width": 0,
            "height": 0,
            "avg_frame_rate": "",
            "nb_read_packets": 0,
            "parts": 0,
            "wall_seconds": 0.0,
            "preflight": asdict(pf),
            "error": "",
        }
        if pf.alive:
            out_path = output_dir / f"{sanitize_filename(cam.name)}_{started_utc:%Y%m%dT%H%M%SZ}.mp4"
            live_jobs[cam.name] = (cam, out_path)
        else:
            clip_entry["error"] = pf.reason
            logging.warning("%s: DEAD (%s)", cam.name, pf.reason)
        clips.append(clip_entry)

    if live_jobs:
        output_dir.mkdir(parents=True, exist_ok=True)

        def request_stop(signum, _frame):
            if not STOP_EVENT.is_set():
                logging.info("Stop requested (signal %s); finalizing recordings...", signum)
            STOP_EVENT.set()

        stop_signals = [signal.SIGINT] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])
        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in stop_signals}
        if args.duration <= 0:
            logging.info("Recording until stopped; press Ctrl+C to finish")
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(live_jobs))) as executor:
                futures = {
                    executor.submit(
                        record_clip, cam, args.duration, out_path, parts_dir, args.max_part_seconds
                    ): cam.name
                    for cam, out_path in live_jobs.values()
                }
                results: dict[str, RecordResult] = {}
                pending = set(futures)
                while pending:
                    # Short timeout keeps the main thread responsive to the stop signal handler.
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = futures[future]
                        try:
                            results[name] = future.result()
                        except Exception as exc:  # noqa: BLE001
                            results[name] = RecordResult(status="FAIL", error=str(exc))
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

        for entry in clips:
            name = entry["cctv_name"]
            if name not in results:
                continue
            rec = results[name]
            entry["parts"] = rec.parts
            entry["wall_seconds"] = round(rec.wall_seconds, 1)
            if rec.status != "OK":
                entry["status"] = "FAIL"
                entry["error"] = rec.error
                continue
            verdict = verify_clip(Path(rec.output_file), args.duration, args.tolerance)
            entry["status"] = verdict["status"]
            entry["output_file"] = rec.output_file
            entry["duration_seconds"] = verdict["duration_seconds"]
            entry["size_bytes"] = verdict["size_bytes"]
            entry["codec_name"] = verdict["codec_name"]
            entry["width"] = verdict["width"]
            entry["height"] = verdict["height"]
            entry["avg_frame_rate"] = verdict["avg_frame_rate"]
            entry["nb_read_packets"] = verdict["nb_read_packets"]
            entry["error"] = verdict["error"]

        if not args.keep_parts:
            for cam, out_path in live_jobs.values():
                stem = sanitize_filename(cam.name)
                for pattern in (f"{stem}_part*.ts", f"{stem}_concat.txt"):
                    for leftover in parts_dir.glob(pattern):
                        leftover.unlink(missing_ok=True)

    n_pass = sum(1 for c in clips if c["status"] == "PASS")
    n_fail = sum(1 for c in clips if c["status"] == "FAIL")
    n_dead = sum(1 for c in clips if c["status"] == "DEAD")

    print(f"{'NAME':<24} {'STATUS':<6} {'DURATION_S':<11} {'SIZE_MB':<9} {'PARTS':<6} {'WALL_S':<8} REASON")
    for c in clips:
        duration_s = f"{c['duration_seconds']:.1f}" if c["duration_seconds"] is not None else "-"
        size_mb = f"{c['size_bytes'] / 1_000_000:.2f}" if c["size_bytes"] else "-"
        print(
            f"{c['cctv_name']:<24} {c['status']:<6} {duration_s:<11} {size_mb:<9} "
            f"{c['parts']:<6} {c['wall_seconds']:<8} {c['error']}"
        )
    print(f"{n_pass}/{len(clips)} PASS, {n_fail} FAIL, {n_dead} DEAD")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"record_report_{started_utc:%Y%m%dT%H%M%SZ}.json"
    report = {
        "generated_at_utc": started_utc.isoformat(),
        "duration_target_seconds": args.duration,
        "tolerance_seconds": args.tolerance,
        "source": "api",
        "clips": clips,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logging.info("Report written to %s", report_path)

    if n_pass == len(clips):
        return 0
    if n_pass > 0:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
