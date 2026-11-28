"""Ekspor .pt -> ONNX FP16 dynamic tanpa NMS -> engine TensorRT.

    .venv\\Scripts\\python.exe -m pkji.export_model

Engine hanya berlaku untuk GPU dan versi TensorRT tempat ia dibangun.
Pembuatan engine diadaptasi dari thesis/extractions/script/build_engine.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .config import DEFAULT_ENGINE_PATH, DEFAULT_IMGSZ, DEFAULT_PT_PATH


def export_onnx(pt_path: Path, onnx_path: Path, imgsz: int) -> dict:
    """Mengembalikan metadata engine (task, names, stride, ...)."""
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    model = YOLO(str(pt_path))
    metadata = {
        "task": model.task,
        "names": dict(model.names),
        "stride": int(model.model.stride.max()),
        "batch": 1,
        "imgsz": [imgsz, imgsz],
        "end2end": False,
        "args": {"dynamic": True, "half": True},
    }
    precision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, dynamic=True, nms=False, end2end=False,
                     simplify=True, batch=1, device=0, **precision)
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), onnx_path)
    return metadata


def build_engine(onnx_path: Path, engine_path: Path, metadata: dict, imgsz: int,
                 batch: int, max_batch: int, workspace_gb: float) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Gagal parse ONNX: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    # H/W dan batch dinamis; dioptimalkan untuk input persegi (default pipeline)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        network.get_input(0).name,
        min=(1, 3, 32, 32),
        opt=(batch, 3, imgsz, imgsz),
        max=(max_batch, 3, imgsz, imgsz),
    )
    config.add_optimization_profile(profile)
    started = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Build engine TensorRT gagal")

    header = json.dumps({**metadata, "batch": max_batch}).encode()
    tmp = engine_path.with_suffix(".engine.tmp")
    with tmp.open("wb") as fh:
        fh.write(len(header).to_bytes(4, "little", signed=True))
        fh.write(header)
        fh.write(engine)
    os.replace(tmp, engine_path)
    print(f"Engine selesai dalam {time.time() - started:.0f}s -> {engine_path} "
          f"({engine_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ekspor model ke TensorRT FP16")
    ap.add_argument("--pt", type=Path, default=DEFAULT_PT_PATH)
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE_PATH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=24, help="Batch yang dioptimalkan")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--workspace", type=float, default=2.0, help="GiB")
    args = ap.parse_args()
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    onnx_path = args.engine.with_suffix(".fp16.onnx")
    metadata = export_onnx(args.pt, onnx_path, args.imgsz)
    print(f"ONNX FP16 -> {onnx_path}")
    build_engine(onnx_path, args.engine, metadata, args.imgsz, args.batch, args.max_batch,
                 args.workspace)


if __name__ == "__main__":
    main()
