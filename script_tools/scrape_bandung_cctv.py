#!/usr/bin/env python3
"""
Scrape CCTV metadata from Bandung's Pelindung site and capture short samples.

Dependencies:
    pip install requests

System dependency:
    Install ffmpeg and make sure the `ffmpeg` executable is on PATH.

    Windows:
        winget install Gyan.FFmpeg
        # or download from https://www.gyan.dev/ffmpeg/builds/

    macOS:
        brew install ffmpeg

    Ubuntu/Debian:
        sudo apt-get update && sudo apt-get install -y ffmpeg
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None
    HTTPAdapter = None
    Retry = None


API_URL = "https://pelindung.bandung.go.id:8443/api/cek"
CSV_PATH = Path("../csv/cctv_bandung_data.csv")
VALIDATED_CSV = Path("../csv/cctv_bandung_data_validated.csv")
SAMPLES_DIR = Path("../cctv_video_samples")
CSV_HEADERS = [
    "CCTV_Name",
    "Location",
    "Provider",
    "m3u8_URL",
    "Sample_Downloaded",
    "Sample_File",
    "Sample_Size_Bytes",
    "Sample_Error",
    "Validated_At_UTC",
]


@dataclass(frozen=True)
class CCTV:
    name: str
    location: str
    provider: str
    stream_url: str


@dataclass(frozen=True)
class SampleStatus:
    downloaded: bool
    file_path: str = ""
    size_bytes: int = 0
    error: str = ""
    validated_at_utc: str = ""


def request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (compatible; BandungCCTVCollector/1.0; "
            "+https://pelindung.bandung.go.id/)"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://pelindung.bandung.go.id/",
    }


def build_session(timeout_retries: int = 3):
    if requests is None:
        raise RuntimeError("requests is not installed")

    session = requests.Session()
    retry = Retry(
        total=timeout_retries,
        connect=timeout_retries,
        read=timeout_retries,
        status=timeout_retries,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(request_headers())
    return session


def normalize_stream_url(raw_url: object) -> str:
    url = str(raw_url or "").strip()
    if url.startswith("https:/") and not url.startswith("https://"):
        url = "https://" + url[len("https:/") :]
    if url.startswith("http:/") and not url.startswith("http://"):
        url = "http://" + url[len("http:/") :]
    return url


def infer_provider(item: dict, stream_url: str) -> str:
    provider = str(item.get("dinas") or "").strip()
    if provider and provider.lower() != "as":
        return provider

    parsed = urlparse(stream_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "video":
        return parts[1].strip() or provider or "UNKNOWN"
    return provider or "UNKNOWN"


def extract_location(item: dict) -> str:
    for key in ("location", "lokasi", "alamat", "address"):
        value = item.get(key)
        if value:
            return str(value).strip()

    lat = str(item.get("lat") or "").strip()
    lng = str(item.get("lng") or "").strip()
    if lat and lng:
        return f"{lat},{lng}"
    return ""


def parse_cctv_items(payload: list[dict]) -> list[CCTV]:
    cctvs: list[CCTV] = []
    seen_urls: set[str] = set()

    for item in payload:
        name = str(item.get("cctv_name") or item.get("name") or "").strip()
        stream_url = normalize_stream_url(item.get("stream_cctv") or item.get("stream_url"))
        if not name or not stream_url:
            logging.warning("Skipping incomplete CCTV record: %r", item)
            continue

        if stream_url in seen_urls:
            logging.info("Skipping duplicate stream URL for %s", name)
            continue
        seen_urls.add(stream_url)

        cctvs.append(
            CCTV(
                name=name,
                location=extract_location(item),
                provider=infer_provider(item, stream_url),
                stream_url=stream_url,
            )
        )

    return cctvs


def fetch_cctv_data(api_url: str, request_timeout: float) -> list[CCTV]:
    logging.info("Fetching CCTV metadata from %s", api_url)
    if requests is not None:
        session = build_session()
        response = session.get(api_url, timeout=request_timeout)
        response.raise_for_status()
        payload = response.json()
    else:
        logging.warning("requests is not installed; using urllib fallback")
        request = Request(api_url, headers=request_headers(), method="GET")
        with urlopen(request, timeout=request_timeout) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list from API, got {type(payload).__name__}")

    cctvs = parse_cctv_items(payload)
    logging.info("Parsed %d CCTV records", len(cctvs))
    return cctvs


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_csv(
    cctvs: Iterable[CCTV],
    csv_path: Path,
    statuses: dict[int, SampleStatus] | None = None,
) -> None:
    statuses = statuses or {}
    try:
        handle = csv_path.open("w", newline="", encoding="utf-8")
        final_path = csv_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = csv_path.with_name(f"{csv_path.stem}_validated_{timestamp}{csv_path.suffix}")
        logging.warning("%s is locked; writing validated CSV to %s", csv_path, final_path)
        handle = final_path.open("w", newline="", encoding="utf-8")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for index, cctv in enumerate(cctvs):
            status = statuses.get(index, SampleStatus(False))
            writer.writerow(
                {
                    "CCTV_Name": cctv.name,
                    "Location": cctv.location,
                    "Provider": cctv.provider,
                    "m3u8_URL": cctv.stream_url,
                    "Sample_Downloaded": "YES" if status.downloaded else "NO",
                    "Sample_File": status.file_path,
                    "Sample_Size_Bytes": status.size_bytes or "",
                    "Sample_Error": status.error,
                    "Validated_At_UTC": status.validated_at_utc,
                }
            )
    logging.info("Wrote CSV to %s", final_path)


def sanitize_filename(name: str, fallback: str = "cctv") -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[^a-z0-9]+", "_", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    return ascii_name or fallback


def sample_output_paths(cctvs: list[CCTV], directory: Path) -> dict[int, Path]:
    seen: dict[str, int] = {}
    paths: dict[int, Path] = {}
    for index, cctv in enumerate(cctvs):
        stem = sanitize_filename(cctv.name)
        seen[stem] = seen.get(stem, 0) + 1
        suffix = "" if seen[stem] == 1 else f"_{seen[stem]}"
        paths[index] = directory / f"{stem}{suffix}.mp4"
    return paths


def validate_sample_file(path: Path, error: str = "") -> SampleStatus:
    validated_at = current_utc_timestamp()
    if path.exists() and path.is_file():
        size = path.stat().st_size
        if size > 0:
            return SampleStatus(True, str(path), size, "", validated_at)
        return SampleStatus(False, str(path), 0, "Sample file exists but is empty", validated_at)
    return SampleStatus(False, str(path), 0, error or "Sample file not found", validated_at)


def validate_existing_samples(paths: dict[int, Path]) -> dict[int, SampleStatus]:
    return {index: validate_sample_file(path) for index, path in paths.items()}


def is_m3u8_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".m3u8")


def capture_video_sample(
    cctv: CCTV,
    output_path: Path,
    duration: int,
    ffmpeg_timeout: int,
) -> SampleStatus:
    if not is_m3u8_url(cctv.stream_url):
        return validate_sample_file(output_path, f"Skipped non-.m3u8 URL: {cctv.stream_url}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".part.mp4")
    if temp_path.exists():
        temp_path.unlink()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-rw_timeout",
        str(ffmpeg_timeout * 1_000_000),
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "2",
        "-i",
        cctv.stream_url,
        "-t",
        str(duration),
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=ffmpeg_timeout + duration + 10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if temp_path.exists():
            temp_path.unlink()
        return validate_sample_file(output_path, f"ffmpeg timed out after {exc.timeout}s")
    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink()
        return validate_sample_file(output_path, f"Could not run ffmpeg: {exc}")

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        error = (completed.stderr or completed.stdout or "").strip()
        return validate_sample_file(output_path, f"ffmpeg failed in {elapsed:.1f}s: {error[:500]}")

    temp_path.replace(output_path)
    return validate_sample_file(output_path)


def capture_samples(
    cctvs: list[CCTV],
    samples_dir: Path,
    duration: int,
    workers: int,
    ffmpeg_timeout: int,
) -> dict[int, SampleStatus]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    paths = sample_output_paths(cctvs, samples_dir)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error("ffmpeg was not found on PATH; skipping video sampling.")
        return validate_existing_samples(paths)

    statuses = validate_existing_samples(paths)
    jobs = [(index, cctv, paths[index]) for index, cctv in enumerate(cctvs) if cctv.stream_url]

    logging.info("Capturing %d video samples with %d workers", len(jobs), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(capture_video_sample, cctv, path, duration, ffmpeg_timeout): (index, cctv)
            for index, cctv, path in jobs
        }
        for future in as_completed(future_map):
            index, cctv = future_map[future]
            status = future.result()
            statuses[index] = status
            if status.downloaded:
                logging.info("Sample saved for %s: %s", cctv.name, status.file_path)
            else:
                logging.warning("Sample failed for %s: %s", cctv.name, status.error)

    return statuses


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Bandung Pelindung CCTV metadata and save 5-second HLS samples."
    )
    parser.add_argument("--api-url", default=API_URL, help="CCTV JSON API URL.")
    parser.add_argument("--csv", default=str(CSV_PATH), help="Output CSV path.")
    parser.add_argument("--samples-dir", default=str(SAMPLES_DIR), help="Output sample directory.")
    parser.add_argument("--duration", type=int, default=5, help="Sample duration in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ffmpeg downloads.")
    parser.add_argument("--request-timeout", type=float, default=20.0, help="API request timeout.")
    parser.add_argument("--ffmpeg-timeout", type=int, default=20, help="Per-stream ffmpeg I/O timeout.")
    parser.add_argument("--limit", type=int, default=0, help="Limit cameras for testing; 0 means all.")
    parser.add_argument("--skip-video", action="store_true", help="Only write the CSV.")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing sample files and update CSV without running ffmpeg.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cctvs = fetch_cctv_data(args.api_url, args.request_timeout)
    except Exception:
        logging.exception("Could not fetch or parse CCTV metadata")
        return 1

    if args.limit > 0:
        cctvs = cctvs[: args.limit]
        logging.info("Limited run to %d CCTV records", len(cctvs))

    samples_dir = Path(args.samples_dir)
    paths = sample_output_paths(cctvs, samples_dir)
    statuses = validate_existing_samples(paths)

    if not args.skip_video and not args.validate_existing:
        workers = max(1, min(args.workers, 8))
        statuses = capture_samples(
            cctvs=cctvs,
            samples_dir=samples_dir,
            duration=max(1, args.duration),
            workers=workers,
            ffmpeg_timeout=max(5, args.ffmpeg_timeout),
        )

    write_csv(cctvs, Path(args.csv), statuses)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
