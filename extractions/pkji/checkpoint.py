"""Checkpoint tahap fitur: state disimpan setelah tiap video selesai.

Checkpoint hanya dipakai bila metode, model, ROI, kode pkji, dan isi cache
deteksi video-video yang sudah diproses masih sama persis.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List

SUBDIR = "checkpoints"


def _source_digest() -> bytes:
    digest = hashlib.sha1()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.digest()


def clear_checkpoints(cache_dir: Path) -> None:
    for path in (cache_dir / SUBDIR).glob("*.pkl"):
        path.unlink(missing_ok=True)


class FeatureCheckpoint:
    def __init__(self, cache_dir: Path, name: str, manifest: dict, roi: bytes) -> None:
        digest = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode())
        digest.update(roi)
        digest.update(_source_digest())
        self.path = cache_dir / SUBDIR / f"{name}-{digest.hexdigest()[:12]}.pkl"

    @staticmethod
    def entry(cache_file: Path) -> tuple:
        return cache_file.name, hashlib.sha1(cache_file.read_bytes()).hexdigest()

    def load(self, videos: List[Path], cache_files: dict, wait):
        """(entri video yang sudah diproses, state); ([], None) bila tidak berlaku."""
        if not self.path.is_file():
            return [], None
        try:
            # file buatan sendiri di folder cache lokal
            data = pickle.loads(self.path.read_bytes())
            done = data["done"]
            if len(done) > len(videos):
                return [], None
            for video, saved in zip(videos, done):
                wait(video)
                if self.entry(cache_files[video]) != saved:
                    return [], None
        except (pickle.UnpicklingError, EOFError, KeyError, AttributeError, ImportError) as exc:
            print(f"[WARN] Checkpoint fitur tidak terbaca, dihitung ulang: {exc}", flush=True)
            return [], None
        return done, data["state"]

    def save(self, done: list, state) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"done": done, "state": state}))
        os.replace(tmp, self.path)
