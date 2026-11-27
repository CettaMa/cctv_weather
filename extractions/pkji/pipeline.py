"""Tahap deteksi: N worker decode + 1-2 inference server GPU lewat shared memory.

Pola diambil dari thesis/extractions/script/feature_extractor.py. Dengan
workers=0 semuanya berjalan di satu proses (untuk debugging dan baseline).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import cache
from .video import VideoReader

INFER_RESPONSE_TIMEOUT_SEC = 300
SERVER_START_TIMEOUT_SEC = 600
INFLIGHT_BATCHES = 2
CHECKPOINT_SECONDS = 15  # jarak checkpoint berkala (jaga-jaga bila proses mati paksa)
DECODE_ERROR = "Frame gagal di-decode"


@dataclass(frozen=True)
class Job:
    video: str
    cache_file: str


class LocalClient:
    """Detector di proses yang sama."""

    def __init__(self, detector) -> None:
        self.detector = detector
        self.names = detector.names

    def submit(self, frames: List[np.ndarray]):
        t0 = time.perf_counter()
        result = self.detector.detect(frames)
        return result, (time.perf_counter() - t0) * 1000.0 / len(frames)

    def fetch(self, ticket):
        return ticket


class ShmClient:
    """Mengirim batch ke inference server; frame lewat shared memory."""

    def __init__(self, worker_id: int, req_queue, resp_queue, shm_name: str,
                 slot_bytes: int, slots: int, names: dict) -> None:
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.slot_bytes = slot_bytes
        self.names = names
        self.free = deque(range(slots))
        self.slot_count = slots
        self.next_id = 0
        self.ready = {}  # respons yang tiba di luar urutan (ada 2 server)

    def submit(self, frames: List[np.ndarray]) -> int:
        slots = []
        for frame in frames:
            slot = self.free.popleft()
            np.ndarray(frame.shape, np.uint8, self.shm.buf, slot * self.slot_bytes)[:] = frame
            slots.append(slot)
        self.next_id += 1
        self.req_queue.put((self.worker_id, self.next_id, frames[0].shape, slots))
        return self.next_id

    def fetch(self, ticket: int):
        while ticket not in self.ready:
            try:
                req_id, slots, payload, ms = self.resp_queue.get(timeout=INFER_RESPONSE_TIMEOUT_SEC)
            except queue.Empty:
                raise RuntimeError("Inference server tidak merespons") from None
            self.free.extend(slots)
            self.ready[req_id] = (payload, ms)
        payload, ms = self.ready.pop(ticket)
        if isinstance(payload, str):
            raise RuntimeError(f"Inferensi gagal: {payload}")
        return payload, ms

    def reset(self) -> None:
        """Ambil sisa respons setelah video gagal, supaya slot tidak bocor."""
        with contextlib.suppress(queue.Empty):
            while len(self.free) < self.slot_count:
                _, slots, _, _ = self.resp_queue.get(timeout=30)
                self.free.extend(slots)
        self.ready.clear()

    def close(self) -> None:
        self.shm.close()


def detect_video(job: Job, client, interval: float, batch: int,
                 stop, decode_threads: int = 0) -> Optional[cache.VideoDetections]:
    """Mendeteksi satu video, melanjutkan checkpoint parsial bila ada.

    None bila dihentikan pengguna; kemajuan tersimpan sebagai checkpoint parsial.
    """
    video = Path(job.video)
    cache_file = Path(job.cache_file)
    det = cache.VideoDetections(video_name=video.name, names=dict(client.names))

    frame_idx, pos_sec, infer_ms, boxes = [], [], [], []
    errors_done = 0      # kegagalan decode sampai sampel terakhir yang tercatat
    prior_seconds = 0.0  # waktu proses pada run sebelumnya
    prior = cache.load_partial(cache_file)
    if prior is not None and len(prior.frame_idx):
        frame_idx = prior.frame_idx.tolist()
        pos_sec = prior.pos_sec.tolist()
        infer_ms = prior.infer_ms.tolist()
        boxes = [prior.detections(k) for k in range(len(frame_idx))]
        errors_done = len(prior.errors)
        prior_seconds = len(frame_idx) / max(1e-6, prior.processing_fps)
        print(f"[LANJUT] {video.name} dari frame {frame_idx[-1]}/{prior.frame_count}", flush=True)

    reader = VideoReader(video, interval, decode_threads,
                         resume_after=frame_idx[-1] if frame_idx else -1)
    if not reader.opened:
        det.failed = True
        det.errors.append(f"Gagal membuka file video: {video}")
        return det

    errors_before = errors_done
    pending: List[np.ndarray] = []
    pending_meta: List[Tuple[int, float, int]] = []
    inflight: deque = deque()
    started = time.perf_counter()
    last_checkpoint = started

    def snapshot(errors: int) -> None:
        det.fps = reader.fps
        det.frame_count = reader.frame_count
        det.frames_read = reader.frames_read
        det.width, det.height = reader.width, reader.height
        det.end_sec = reader.end_sec
        det.errors = [DECODE_ERROR] * errors + [e for e in det.errors if e != DECODE_ERROR]
        seconds = prior_seconds + time.perf_counter() - started
        det.processing_fps = len(frame_idx) / max(1e-6, seconds)
        det.frame_idx = np.asarray(frame_idx, np.int32)
        det.pos_sec = np.asarray(pos_sec, np.float64)
        det.infer_ms = np.asarray(infer_ms, np.float32)
        det.offsets = np.concatenate(([0], np.cumsum([len(b) for b in boxes]))).astype(np.int32)
        det.boxes = np.concatenate(boxes) if boxes else np.zeros((0, 6), np.float32)

    def checkpoint() -> None:
        if frame_idx:
            snapshot(errors_done)
            cache.save(cache.partial_path(cache_file), det)

    def submit() -> None:
        inflight.append((client.submit(list(pending)), list(pending_meta)))
        pending.clear()
        pending_meta.clear()

    def drain(limit: int) -> None:
        nonlocal errors_done
        while len(inflight) > limit:
            ticket, meta = inflight.popleft()
            results, ms = client.fetch(ticket)
            for (idx, pos, errors), dets in zip(meta, results):
                frame_idx.append(idx)
                pos_sec.append(pos)
                infer_ms.append(ms)
                boxes.append(dets)
                errors_done = errors

    try:
        for idx, pos, frame in reader:
            if stop.is_set():
                break
            pending.append(frame)
            pending_meta.append((idx, pos, errors_before + reader.decode_errors))
            if len(pending) == batch:
                drain(INFLIGHT_BATCHES - 1)
                submit()
                if time.perf_counter() - last_checkpoint >= CHECKPOINT_SECONDS:
                    checkpoint()
                    last_checkpoint = time.perf_counter()
        if pending and not stop.is_set():
            drain(INFLIGHT_BATCHES - 1)
            submit()
        drain(0)
    except Exception as exc:
        det.failed = True
        det.errors.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] {video.name}: {type(exc).__name__}: {exc}", flush=True)
        if hasattr(client, "reset"):
            client.reset()
    finally:
        reader.close()
    if stop.is_set():
        if hasattr(client, "reset"):
            client.reset()
        checkpoint()
        return None

    snapshot(errors_before + reader.decode_errors)
    return det


def _report(job: Job, det: cache.VideoDetections) -> str:
    return (
        f"[DETEKSI] {det.video_name} | frame={det.frames_read}/{det.frame_count} | "
        f"sampel={len(det.frame_idx)} | {det.processing_fps:.0f} sampel/s"
        + (" | GAGAL" if det.failed else "")
    )


def _child_init() -> None:
    """Abaikan Ctrl-C (ditangani induk lewat event) dan keluar bila induk mati."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    parent = mp.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _server(settings: dict, req_queue, resp_queues, shm_names, slot_bytes, status_queue) -> None:
    _child_init()
    try:
        cv2.setNumThreads(2)
        from .detector import build_detector

        detector = build_detector(settings)
    except Exception as exc:
        status_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    status_queue.put(("ready", detector.names))
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    try:
        while True:
            item = req_queue.get()
            if item is None:
                break
            worker_id, req_id, shape, slots = item
            try:
                frames = [
                    np.ndarray(shape, np.uint8, shms[worker_id].buf, s * slot_bytes) for s in slots
                ]
                t0 = time.perf_counter()
                result = detector.detect(frames)
                ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
                result = [np.array(r, np.float32) for r in result]
                del frames
            except Exception as exc:
                result, ms = f"{type(exc).__name__}: {exc}", 0.0
            resp_queues[worker_id].put((req_id, slots, result, ms))
    finally:
        for shm in shms:
            with contextlib.suppress(BufferError):
                shm.close()


def _worker(worker_id: int, jobs, done, stop, req_queue, resp_queue, shm_name,
            slot_bytes, slots, names, interval, batch) -> None:
    _child_init()
    cv2.setNumThreads(1)
    client = ShmClient(worker_id, req_queue, resp_queue, shm_name, slot_bytes, slots, names)
    try:
        while not stop.is_set():
            job = jobs.get()
            if job is None:
                break
            try:
                det = detect_video(job, client, interval, batch, stop, decode_threads=1)
                if det is None:
                    break
                cache.save_final(Path(job.cache_file), det)
                done.put((job.video, _report(job, det)))
            except Exception as exc:
                done.put((job.video, f"[ERROR] {job.video}: {type(exc).__name__}: {exc}"))
    finally:
        client.close()


def _max_frame_bytes(jobs: List[Job]) -> int:
    largest = 0
    for job in jobs:
        cap = cv2.VideoCapture(job.video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        largest = max(largest, w * h * 3)
    return largest or 1920 * 1080 * 3


def run_detection(jobs: List[Job], settings: dict, interval: float, workers: int,
                  servers: int, stop, on_done: Callable[[str], None] = lambda video: None) -> None:
    """Menulis satu file cache per job, urut sesuai `jobs`. `stop` = mp.Event.

    `on_done(video)` dipanggil di proses induk begitu satu video selesai.
    """
    if not jobs:
        return
    batch = settings["batch"]
    total = len(jobs)
    if workers <= 0:
        from .detector import build_detector

        client = LocalClient(build_detector(settings))
        for n, job in enumerate(jobs, 1):
            if stop.is_set():
                break
            det = detect_video(job, client, interval, batch, stop)
            if det is None:
                break
            cache.save_final(Path(job.cache_file), det)
            print(f"({n}/{total}) {_report(job, det)}", flush=True)
            on_done(job.video)
        return

    workers = min(workers, total)
    slot_bytes = _max_frame_bytes(jobs)
    slots = batch * INFLIGHT_BATCHES
    print(f"[*] {workers} worker decode + {servers} inference server, batch {batch}", flush=True)

    ctx = mp.get_context("spawn")
    req_queue, done, status, job_queue = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue()
    resp_queues = [ctx.Queue() for _ in range(workers)]
    shms, server_procs, worker_procs = [], [], []
    try:
        shms = [shared_memory.SharedMemory(create=True, size=slot_bytes * slots) for _ in range(workers)]
        shm_names = [s.name for s in shms]
        names = None
        for _ in range(servers):  # satu per satu supaya alokasi VRAM tidak bertabrakan
            proc = ctx.Process(target=_server,
                               args=(settings, req_queue, resp_queues, shm_names, slot_bytes, status))
            proc.start()
            server_procs.append(proc)
            deadline = time.monotonic() + SERVER_START_TIMEOUT_SEC
            while True:
                try:
                    kind, value = status.get(timeout=1)
                    break
                except queue.Empty:
                    if not proc.is_alive() or time.monotonic() > deadline:
                        kind, value = "error", "proses server berhenti saat inisialisasi"
                        break
            if kind == "error":
                raise RuntimeError(f"Inference server gagal start: {value}")
            names = value

        # urut waktu supaya tahap fitur bisa berjalan bersamaan
        for job in jobs:
            job_queue.put(job)
        for i in range(workers):
            job_queue.put(None)
            proc = ctx.Process(
                target=_worker,
                args=(i, job_queue, done, stop, req_queue, resp_queues[i], shm_names[i],
                      slot_bytes, slots, names, interval, batch),
            )
            proc.start()
            worker_procs.append(proc)

        finished = 0
        while any(p.is_alive() for p in worker_procs) or not done.empty():
            try:
                video, message = done.get(timeout=1)
                finished += 1
                print(f"({finished}/{total}) {message}", flush=True)
                on_done(video)
            except queue.Empty:
                if not any(p.is_alive() for p in server_procs):
                    raise RuntimeError("Semua inference server mati") from None
        crashed = [p.exitcode for p in worker_procs if p.exitcode not in (0, None)]
        if crashed and not stop.is_set():
            raise RuntimeError(f"Worker berhenti tidak normal (exit code {crashed})")
    finally:
        for _ in server_procs:
            req_queue.put(None)
        for proc in worker_procs + server_procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
        for shm in shms:
            shm.close()
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


@contextlib.contextmanager
def instance_lock(path: Path):
    """Satu proses ekstraksi per folder cache (mencegah VRAM habis dan cache bentrok)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Ekstraksi lain sedang berjalan pada folder cache ini.") from None
    try:
        yield
    finally:
        handle.close()
