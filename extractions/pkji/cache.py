"""Cache deteksi per video (Parquet). Sekaligus checkpoint untuk resume."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CACHE_VERSION = 1
_META_KEY = b"pkji"


@dataclass
class VideoDetections:
    video_name: str
    fps: float = 25.0
    frame_count: int = 0
    frames_read: int = 0
    width: int = 0
    height: int = 0
    end_sec: float = 0.0
    processing_fps: float = 0.0
    failed: bool = False
    errors: List[str] = field(default_factory=list)
    names: Dict[int, str] = field(default_factory=dict)
    frame_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    pos_sec: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    infer_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # deteksi semua sampel disambung; sampel k = boxes[offsets[k]:offsets[k+1]]
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), np.float32))
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int32))

    def detections(self, k: int) -> np.ndarray:
        return self.boxes[self.offsets[k] : self.offsets[k + 1]]


_META_FIELDS = (
    "video_name", "fps", "frame_count", "frames_read", "width", "height",
    "end_sec", "processing_fps", "failed", "errors",
)


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """Sidik cepat: ukuran + 1 MiB awal dan akhir."""
    size = path.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with path.open("rb") as fh:
        digest.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(chunk, size - chunk))
            digest.update(fh.read(chunk))
    return digest.hexdigest()


def cache_path(cache_dir: Path, video: Path, detect_settings: dict) -> Path:
    payload = json.dumps(
        {"v": CACHE_VERSION, "video": file_fingerprint(video), **detect_settings}, sort_keys=True
    )
    key = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return cache_dir / f"{video.stem}.{key}.parquet"


def save(path: Path, det: VideoDetections) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = pa.array(det.boxes.astype(np.float32).ravel(), type=pa.float32())
    boxes = pa.ListArray.from_arrays(pa.array(det.offsets.astype(np.int32) * 6), flat)
    table = pa.table(
        {
            "frame_idx": pa.array(det.frame_idx, type=pa.int32()),
            "pos_sec": pa.array(det.pos_sec, type=pa.float64()),
            "infer_ms": pa.array(det.infer_ms, type=pa.float32()),
            "boxes": boxes,
        }
    )
    meta = {name: getattr(det, name) for name in _META_FIELDS}
    meta["names"] = {str(k): v for k, v in det.names.items()}
    table = table.replace_schema_metadata({_META_KEY: json.dumps(meta).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def partial_path(path: Path) -> Path:
    """Checkpoint deteksi video yang belum selesai."""
    return path.with_suffix(".partial.parquet")


def load_partial(path: Path) -> Optional[VideoDetections]:
    partial = partial_path(path)
    if not partial.is_file():
        return None
    try:
        return load(partial)
    except Exception as exc:
        print(f"[WARN] Checkpoint {partial.name} rusak, video diulang: {exc}", flush=True)
        return None


def save_final(path: Path, det: VideoDetections) -> None:
    """Simpan hasil akhir; checkpoint parsial dibuang hanya bila video berhasil."""
    save(path, det)
    if not det.failed:
        partial_path(path).unlink(missing_ok=True)


def load_meta(path: Path) -> dict:
    import pyarrow.parquet as pq

    return json.loads(pq.read_schema(path).metadata[_META_KEY])


def load(path: Path) -> VideoDetections:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[_META_KEY])
    names = {int(k): v for k, v in meta.pop("names").items()}
    boxes = table.column("boxes").combine_chunks()
    offsets = boxes.offsets.to_numpy()
    flat = boxes.values.to_numpy(zero_copy_only=False)[offsets[0] : offsets[-1]]
    return VideoDetections(
        **meta,
        names=names,
        frame_idx=table.column("frame_idx").to_numpy(),
        pos_sec=table.column("pos_sec").to_numpy(),
        infer_ms=table.column("infer_ms").to_numpy(),
        boxes=flat.astype(np.float32).reshape(-1, 6),
        offsets=((offsets - offsets[0]) // 6).astype(np.int32),
    )


def is_reusable(path: Path) -> bool:
    """Cache dipakai ulang hanya bila ada, terbaca, dan videonya tidak gagal diproses."""
    if not path.is_file():
        return False
    try:
        return not load_meta(path)["failed"]
    except Exception:
        return False
