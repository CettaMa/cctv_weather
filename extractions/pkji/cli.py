"""Antarmuka baris perintah: deteksi (GPU, ter-cache) lalu hitung fitur."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from . import cache
from .aggregate import write_multi_interval_summaries
from .checkpoint import FeatureCheckpoint, clear_checkpoints
from .common import collect_video_files, open_csv_writer, parse_video_metadata
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIDENCE,
    DEFAULT_CPU_THREADS,
    DEFAULT_ENGINE_PATH,
    DEFAULT_IMGSZ,
    DEFAULT_INFERENCE_INTERVAL_SECONDS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_ONNX_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_ROI_CONFIG_PATH,
    LEGACY_INFERENCE_INTERVAL_SECONDS,
    MAX_DETECTIONS,
    load_params,
    load_roi,
    resolve_roi_path,
)
from .features import FeatureExtractor
from .pipeline import Job, instance_lock, run_detection
from .schemas import SUBSESSION_CSV_FIELDS, VIDEO_SUMMARY_CSV_FIELDS

MANIFEST_NAME = "run_manifest.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pkji", description="Ekstraksi DJ & metrik PKJI 2023 dari video CCTV"
    )
    add = parser.add_argument
    add("--input", "-i", type=Path, required=True, help="1 file video atau folder video.")
    add("--recursive", action="store_true", help="Cari video hingga ke subfolder.")
    add("--output-dir", "-o", type=Path, default=DEFAULT_RESULTS_DIR)
    add("--roi", type=Path, default=DEFAULT_ROI_CONFIG_PATH,
        help="File ROI, atau folder berisi <LOKASI>.json.")
    add("--legacy", action="store_true",
        help="Pakai cara hitung script lama (tracker, arus, timestamp, agregasi, status).")
    add("--backend", choices=["auto", "tensorrt", "onnx"], default="auto",
        help="auto = TensorRT bila engine ada, selain itu ONNX Runtime.")
    add("--model", type=Path, default=None, help="Path .engine/.pt (tensorrt) atau .onnx.")
    add("--device", choices=["cuda", "auto", "cpu"], default="cuda", help="Hanya backend onnx.")
    add("--gpu-id", type=int, default=0)
    add("--threads", type=int, default=DEFAULT_CPU_THREADS, help="Thread CPU backend onnx.")
    add("--imgsz", type=int, default=DEFAULT_IMGSZ)
    add("--shape", choices=["square", "rect"], default="square",
        help="rect (480x640) lebih cepat tetapi gagal uji paritas: deteksi bis turun sekitar 50%%.")
    add("--conf", type=float, default=DEFAULT_CONFIDENCE)
    add("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    add("--inference-interval", type=float, default=None,
        help=f"Jarak antar sampel dalam detik (default {DEFAULT_INFERENCE_INTERVAL_SECONDS}; "
             f"{LEGACY_INFERENCE_INTERVAL_SECONDS} dengan --legacy).")
    add("--workers", type=int, default=4, help="Proses decode paralel; 0 = satu proses.")
    add("--servers", type=int, default=1,
        help="Inference server GPU. 2 hanya sedikit lebih cepat dan memakai VRAM 2x.")
    add("--batch", type=int, default=24, help="Ukuran batch inferensi.")
    add("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    add("--redetect", action="store_true", help="Abaikan cache deteksi dan deteksi ulang.")
    add("--append", action="store_true", help="Tambahkan baris ke CSV yang sudah ada.")
    add("--no-multi-interval", action="store_true", help="Lewati agregasi 5-60 menit.")
    args = parser.parse_args(argv)
    if args.inference_interval is None:
        args.inference_interval = (LEGACY_INFERENCE_INTERVAL_SECONDS if args.legacy
                                   else DEFAULT_INFERENCE_INTERVAL_SECONDS)
    if args.inference_interval <= 0:
        parser.error("--inference-interval harus lebih besar dari 0.")
    if not 1 <= args.servers <= 2:
        parser.error("--servers harus 1 atau 2 (3 konteks CUDA di GPU 6 GB thrashing).")
    if not 1 <= args.batch <= 32:
        parser.error("--batch harus 1..32.")
    return args


def detector_settings(args) -> dict:
    backend = args.backend
    model = args.model
    if backend == "auto":
        if model is not None:
            backend = "onnx" if model.suffix.lower() == ".onnx" else "tensorrt"
        else:
            backend = "tensorrt" if DEFAULT_ENGINE_PATH.is_file() else "onnx"
    if model is None:
        model = DEFAULT_ENGINE_PATH if backend == "tensorrt" else DEFAULT_ONNX_PATH
    if not model.is_file():
        raise FileNotFoundError(f"File model tidak ditemukan: {model}")
    rect = args.shape == "rect"
    return {
        "backend": backend, "model": str(model.resolve()), "device": args.device,
        "gpu_id": args.gpu_id, "cpu_threads": args.threads, "imgsz": args.imgsz,
        "rect": rect, "conf": args.conf, "iou": args.iou, "batch": args.batch,
    }


def build_manifest(args, settings: dict) -> dict:
    return {
        "method": "legacy" if args.legacy else "default",
        "tracker": "centroid" if args.legacy else "motion-v1",
        "backend": settings["backend"],
        "model": Path(settings["model"]).name,
        "model_sha1": cache.file_fingerprint(Path(settings["model"])),
        "imgsz": settings["imgsz"],
        "rect": settings["rect"],
        "conf": settings["conf"],
        "iou": settings["iou"],
        "inference_interval": args.inference_interval,
    }


def check_append(output_dir: Path, manifest: dict) -> None:
    path = output_dir / MANIFEST_NAME
    if not path.is_file():
        if (output_dir / "video_summary_pkji2.csv").exists():
            raise RuntimeError(
                "--append ditolak: CSV lama tanpa run_manifest.json (metodenya tidak diketahui)."
            )
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    changed = sorted(k for k in manifest if old.get(k) != manifest[k])
    if changed:
        raise RuntimeError(f"--append ditolak: metode berbeda dari hasil lama ({', '.join(changed)}).")


class _Aborted(Exception):
    """Tahap fitur dibatalkan karena deteksi tidak selesai."""


def write_csv_atomic(path: Path, fields: List[str], rows: List[dict], append: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if append and path.exists():
        shutil.copyfile(path, tmp)
    elif tmp.exists():
        tmp.unlink()
    handle, writer = open_csv_writer(tmp, fields, append=append)
    with handle:
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if not k.startswith("_")})
    os.replace(tmp, path)


def extract_features(args, videos: List[Path], cache_files: dict, wait=lambda video: None,
                     check=lambda: None, manifest: dict = None):
    """Mengembalikan (baris video, [(lokasi, params, baris menit)]).

    `wait(video)` memblokir sampai cache deteksi video tersebut siap. Dengan
    `manifest`, state disimpan setelah tiap video sehingga bisa dilanjutkan.
    """
    roi_cache = {}

    def make_extractor(location: str) -> FeatureExtractor:
        roi_path = resolve_roi_path(args.roi, location)
        params = load_params(roi_path)

        def roi_loader(width: int, height: int):
            key = (roi_path, width, height)
            if key not in roi_cache:
                roi_cache[key] = load_roi(roi_path, width, height)
            return roi_cache[key]

        return FeatureExtractor(location, params, roi_loader, args.inference_interval,
                                args.legacy)

    def open_checkpoint(name: str, locations: List[str]):
        if manifest is None:
            return None
        paths = sorted({resolve_roi_path(args.roi, location) for location in locations})
        roi = b"".join(path.read_bytes() for path in paths)
        return FeatureCheckpoint(args.cache_dir, name, manifest, roi)

    def resume(store, group: List[Path]):
        """(jumlah video yang sudah selesai, entri-entrinya, state)."""
        if store is None:
            return 0, [], None
        done, state = store.load(group, cache_files, wait)
        if done:
            print(f"[LANJUT] tahap fitur: {len(done)} video dari checkpoint", flush=True)
        return len(done), done, state

    video_rows: List[dict] = []
    minute_groups = []
    if args.legacy:  # urutan waktu global, tiap video berdiri sendiri
        rows, first_params = [], None
        store = open_checkpoint("legacy", [parse_video_metadata(v)[0] for v in videos])
        count, done, state = resume(store, videos)
        if count:
            video_rows, rows, first_params = state
        for video in videos[count:]:
            location, start = parse_video_metadata(video)
            extractor = make_extractor(location)
            first_params = first_params or extractor.p
            wait(video)
            extractor.add_video(cache.load(cache_files[video]), start, check)
            video_rows += extractor.video_rows
            rows += extractor.minute_rows
            if store is not None:
                done.append(store.entry(cache_files[video]))
                store.save(done, (video_rows, rows, first_params))
        minute_groups.append(("", first_params, rows))
        return video_rows, minute_groups

    by_location = {}
    for video in videos:
        location, start = parse_video_metadata(video)
        by_location.setdefault(location, []).append((start, video))
    for location, items in by_location.items():
        extractor = make_extractor(location)
        store = open_checkpoint(location, [location])
        count, done, state = resume(store, [video for _, video in items])
        if count:
            extractor.restore(state)
        for start, video in items[count:]:
            wait(video)
            extractor.add_video(cache.load(cache_files[video]), start, check)
            if store is not None:
                done.append(store.entry(cache_files[video]))
                store.save(done, extractor.snapshot())
        extractor.finish()
        video_rows += extractor.video_rows
        minute_groups.append((location, extractor.p, extractor.minute_rows))
    video_rows.sort(key=lambda r: (r["video_start_local"], r["video_file"]))
    return video_rows, minute_groups


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # jangan instal paket diam-diam
    started = time.perf_counter()
    print("Tekan Ctrl+C untuk pause (checkpoint disimpan).", flush=True)

    videos = collect_video_files(args.input, recursive=args.recursive)
    if not videos:
        raise FileNotFoundError(f"Tidak ada file video ditemukan di: {args.input}")
    for location in {parse_video_metadata(v)[0] for v in videos}:
        load_params(resolve_roi_path(args.roi, location))  # gagal cepat sebelum deteksi

    settings = detector_settings(args)
    manifest = build_manifest(args, settings)
    if args.append:
        check_append(args.output_dir, manifest)

    key = {k: manifest[k] for k in
           ("backend", "model_sha1", "imgsz", "rect", "conf", "iou", "inference_interval")}
    key["max_det"] = MAX_DETECTIONS
    cache_files = {v: cache.cache_path(args.cache_dir, v, key) for v in videos}
    jobs = [Job(str(v), str(c)) for v, c in cache_files.items()
            if args.redetect or not cache.is_reusable(c)]

    print(f"Video                : {len(videos)} ({len(videos) - len(jobs)} dari cache)", flush=True)
    print(f"Backend / model      : {settings['backend']} / {settings['model']}", flush=True)
    print(f"Metode perhitungan   : {manifest['method']}", flush=True)
    print(f"Output / cache       : {args.output_dir} / {args.cache_dir}", flush=True)

    stop = mp.get_context("spawn").Event()

    def request_stop(signum, _frame) -> None:
        stop.set()
        print(f"\nSignal {signum} diterima. Menyimpan checkpoint lalu berhenti...", flush=True)

    # tahap fitur berjalan di thread terpisah, mengikuti deteksi
    ready = {str(v): threading.Event() for v in videos}
    pending = {job.video for job in jobs}
    for name, event in ready.items():
        if name not in pending:
            event.set()
    abort = threading.Event()

    def wait(video: Path) -> None:
        while not ready[str(video)].wait(0.5):
            if abort.is_set() or stop.is_set():
                raise _Aborted
        if stop.is_set() or not cache_files[video].is_file():
            raise _Aborted

    def check() -> None:
        if stop.is_set():
            raise _Aborted

    result = None
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break
        signals.append(signal.SIGBREAK)
    previous = {s: signal.signal(s, request_stop) for s in signals}
    try:
        with instance_lock(args.cache_dir / ".lock"):
            if args.redetect:
                for job in jobs:
                    Path(job.cache_file).unlink(missing_ok=True)
                    cache.partial_path(Path(job.cache_file)).unlink(missing_ok=True)
                clear_checkpoints(args.cache_dir)
            with ThreadPoolExecutor(max_workers=1) as pool:
                features = pool.submit(extract_features, args, videos, cache_files, wait,
                                       check, manifest)
                try:
                    run_detection(jobs, settings, args.inference_interval, args.workers,
                                  args.servers, stop, on_done=lambda video: ready[video].set())
                    detect_seconds = time.perf_counter() - started
                finally:
                    abort.set()  # video yang belum siap tidak akan pernah siap
                try:
                    # tidur singkat (bukan result() langsung) agar Ctrl+C tetap diterima
                    while not features.done():
                        time.sleep(0.2)
                    result = features.result()
                except _Aborted:
                    pass
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    missing = [v for v, c in cache_files.items() if not c.is_file()]
    if stop.is_set() or missing or result is None:
        partial = sum(cache.partial_path(cache_files[v]).is_file() for v in missing)
        print(
            f"[{'PAUSE' if stop.is_set() else 'STOP'}] Checkpoint tersimpan: "
            f"{len(videos) - len(missing)}/{len(videos)} video terdeteksi, {partial} video "
            "sebagian. CSV ditulis setelah semuanya selesai.\n"
            "Lanjutkan dengan menjalankan ulang perintah yang sama"
            + (" TANPA --redetect." if args.redetect else "."),
            flush=True,
        )
        return 130 if stop.is_set() else 1

    video_rows, minute_groups = result

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    minute_rows = [row for _, _, rows in minute_groups for row in rows]
    minute_rows_sorted = minute_rows if args.legacy else sorted(
        minute_rows, key=lambda r: (r["one_min_bucket"], r["location"])
    )
    write_csv_atomic(out / "video_summary_pkji2.csv", VIDEO_SUMMARY_CSV_FIELDS, video_rows, args.append)
    write_csv_atomic(out / "subsession_1min_pkji2.csv", SUBSESSION_CSV_FIELDS,
                     minute_rows_sorted, args.append)
    if not args.no_multi_interval:
        for location, params, rows in minute_groups:
            # lebih dari satu lokasi: file interval dipisah per lokasi
            target = out / location if len(minute_groups) > 1 else out
            write_multi_interval_summaries(rows, target, params, weighted=not args.legacy,
                                           append=args.append)
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    clear_checkpoints(args.cache_dir)  # run selesai, state fitur tidak diperlukan lagi

    failed = [r["video_file"] for r in video_rows if r["status"] != "OK"]
    if failed:
        print(f"[WARN] Status bukan OK: {', '.join(failed)}", flush=True)
    total = time.perf_counter() - started
    print(
        f"\nEkstraksi selesai: {len(minute_rows)} baris menit | deteksi {detect_seconds:.1f}s | "
        f"sisa tahap fitur {total - detect_seconds:.1f}s | total {total:.1f}s",
        flush=True,
    )
    return 0


def run() -> None:
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
