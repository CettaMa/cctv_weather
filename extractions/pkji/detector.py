"""Backend deteksi: ONNX Runtime (model lama / ONNX apa pun) dan TensorRT/PyTorch."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD, MAX_DETECTIONS

NMS_CLASS_OFFSET = 7680.0
PAD_VALUE = 114


def letterbox_geometry(
    orig_h: int, orig_w: int, imgsz: int, rect: bool, stride: int = 32
) -> Tuple[float, int, int, Tuple[int, int, int, int]]:
    """Mengembalikan (skala, lebar, tinggi hasil resize, pad (kiri, kanan, atas, bawah))."""
    scale = min(imgsz / orig_h, imgsz / orig_w)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    dw, dh = imgsz - new_w, imgsz - new_h
    if rect:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2.0, dh / 2.0
    pad = (
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        int(round(dh - 0.1)), int(round(dh + 0.1)),
    )
    return scale, new_w, new_h, pad


def parse_names(raw) -> Dict[int, str]:
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, list):
        return {i: str(v) for i, v in enumerate(parsed)}
    raise ValueError(f"Format daftar kelas tidak dikenal: {raw!r}")


def _to_original(boxes: np.ndarray, scale: float, left: int, top: int, w: int, h: int) -> np.ndarray:
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / scale, 0, w - 1)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / scale, 0, h - 1)
    return boxes


class OnnxDetector:
    """ONNX Runtime. Mendukung output dengan NMS tertanam [N,300,6] maupun output mentah."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        gpu_id: int = 0,
        cpu_threads: int = 4,
        imgsz: int = 640,
        rect: bool = False,
        conf: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls") and device != "cpu":
            try:
                ort.preload_dlls()
            except Exception as exc:
                print(f"[WARN] preload_dlls() dilewati: {exc}", flush=True)

        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        cuda = (
            "CUDAExecutionProvider",
            {
                "device_id": gpu_id,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        )
        if device == "cuda" and not cuda_available:
            raise RuntimeError(
                "CUDAExecutionProvider tidak tersedia. Instal onnxruntime-gpu atau pakai --device auto/cpu."
            )
        providers = (
            [cuda, "CPUExecutionProvider"]
            if device != "cpu" and cuda_available
            else ["CPUExecutionProvider"]
        )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if providers[0] == "CPUExecutionProvider":
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "ONNX Runtime gagal mengaktifkan CUDAExecutionProvider (cek versi CUDA/cuDNN)."
            )

        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_dtype = np.float16 if "float16" in str(meta.type).lower() else np.float32
        self.dynamic_batch = not isinstance(meta.shape[0], int)
        self.rect = rect and not isinstance(meta.shape[2], int)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.output_names = [o.name for o in self.session.get_outputs()]
        out_shape = self.session.get_outputs()[0].shape
        self.embedded_nms = len(out_shape) == 3 and out_shape[2] == 6

        custom = self.session.get_modelmeta().custom_metadata_map
        if "names" not in custom:
            raise ValueError(f"Model {model_path.name} tidak menyimpan metadata 'names' (daftar kelas).")
        self.names = parse_names(custom["names"])

        if self.embedded_nms and (conf < DEFAULT_CONFIDENCE or iou != DEFAULT_IOU_THRESHOLD):
            print(
                "[WARN] Model ini memuat NMS tertanam (conf 0.25, IoU 0.5): --iou dan --conf < 0.25 "
                "tidak berefek. Pakai model tanpa NMS (python -m pkji.export_model).",
                flush=True,
            )
        warm = np.zeros((1, 3, imgsz, imgsz), dtype=self.input_dtype)
        self.session.run(self.output_names, {self.input_name: warm})

    def _blob(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale, new_w, new_h, (left, right, top, bottom) = letterbox_geometry(h, w, self.imgsz, self.rect)
        if (w, h) != (new_w, new_h):
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
        )
        blob = padded[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=self.input_dtype) / np.asarray(255.0, dtype=self.input_dtype)
        return blob, scale, left, top

    def _postprocess(self, pred: np.ndarray) -> np.ndarray:
        """pred untuk satu gambar -> (N,6) di koordinat input model."""
        if self.embedded_nms:
            return pred[pred[:, 4] >= self.conf][:MAX_DETECTIONS].copy()
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        conf = scores[np.arange(len(scores)), class_ids]
        mask = conf > self.conf
        if not np.any(mask):
            return np.zeros((0, 6), np.float32)
        cx, cy, w, h = pred[mask, :4].T
        conf, class_ids = conf[mask], class_ids[mask]
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        offsets = class_ids.astype(np.float32) * NMS_CLASS_OFFSET
        keep = cv2.dnn.NMSBoxes(
            np.stack([x1 + offsets, y1 + offsets, w, h], axis=1).tolist(),
            conf.tolist(), self.conf, self.iou,
        )
        if len(keep) == 0:
            return np.zeros((0, 6), np.float32)
        keep = np.asarray(keep).flatten()[:MAX_DETECTIONS]
        return np.stack(
            [x1, y1, cx + w / 2.0, cy + h / 2.0, conf, class_ids.astype(np.float32)], axis=1
        )[keep].astype(np.float32)

    def detect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        prepared = [self._blob(f) for f in frames]
        if self.dynamic_batch:
            batch = np.stack([p[0] for p in prepared])
            preds = self.session.run(self.output_names, {self.input_name: batch})[0]
        else:
            preds = [
                self.session.run(self.output_names, {self.input_name: p[0][None]})[0][0]
                for p in prepared
            ]
        results = []
        for frame, (_, scale, left, top), pred in zip(frames, prepared, preds):
            dets = self._postprocess(np.asarray(pred, dtype=np.float32))
            h, w = frame.shape[:2]
            results.append(_to_original(dets, scale, left, top, w, h))
        return results


class TorchDetector:
    """TensorRT engine (atau .pt) lewat AutoBackend; preprocess dan NMS di GPU."""

    def __init__(
        self,
        model_path: Path,
        gpu_id: int = 0,
        imgsz: int = 640,
        rect: bool = False,
        conf: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU_THRESHOLD,
        max_batch: int = 32,
    ) -> None:
        import torch
        import torchvision
        from ultralytics.nn.autobackend import AutoBackend

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT/PyTorch backend membutuhkan GPU CUDA.")
        self.torch = torch
        self.nms = torchvision.ops.batched_nms
        self.buffer = None  # buffer pinned untuk upload, dibuat saat batch pertama
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        self.backend = AutoBackend(str(model_path), device=self.device, fp16=True, verbose=False)
        self.fp16 = bool(self.backend.fp16)
        self.names = parse_names(self.backend.names)
        if set(self.names.values()) == {f"class{i}" for i in self.names}:
            raise ValueError(f"Engine {model_path.name} tidak menyimpan metadata kelas.")
        if getattr(self.backend, "end2end", False):
            raise ValueError("Model end2end tidak didukung; ekspor ulang dengan end2end=False.")
        self.imgsz = imgsz
        self.rect = rect
        self.conf = conf
        self.iou = iou
        self.max_batch = max_batch
        self.detect([np.zeros((imgsz * 3 // 4, imgsz, 3), dtype=np.uint8)])  # warmup (frame 4:3)

    def detect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for start in range(0, len(frames), self.max_batch):
            results.extend(self._detect_batch(frames[start : start + self.max_batch]))
        return results

    def _detect_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        torch = self.torch
        h, w = frames[0].shape[:2]
        scale, new_w, new_h, (left, right, top, bottom) = letterbox_geometry(h, w, self.imgsz, self.rect)
        if (w, h) != (new_w, new_h):
            frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for f in frames]
        shape = (self.max_batch, *frames[0].shape)
        if self.buffer is None or self.buffer.shape != shape:
            self.buffer = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        batch = self.buffer[: len(frames)]
        np.stack(frames, out=batch.numpy())
        with torch.inference_mode():
            # blocking: buffer dipakai ulang pada batch berikutnya
            x = batch.to(self.device).permute(0, 3, 1, 2)
            x = torch.nn.functional.pad(x, (left, right, top, bottom), value=PAD_VALUE).flip(1)
            x = (x.half() if self.fp16 else x.float()).div_(255)
            # TensorRT membaca memori mentah: wajib NCHW contiguous
            y = self.backend(x.contiguous())
            y = y[0] if isinstance(y, (list, tuple)) else y
            pred = y.float().transpose(1, 2)  # (B, anchor, 4 + kelas)
            scores, classes = pred[..., 4:].max(-1)
            image_idx, anchor_idx = (scores > self.conf).nonzero(as_tuple=True)
            cxcywh = pred[image_idx, anchor_idx, :4]
            boxes = torch.cat((cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2), 1)
            sc = scores[image_idx, anchor_idx]
            cl = classes[image_idx, anchor_idx]
            # satu panggilan NMS untuk seluruh batch, terpisah per (gambar, kelas)
            groups = image_idx * (pred.shape[2] - 4) + cl
            keep = self.nms(boxes, sc, groups, self.iou)  # urut skor menurun
            out = torch.cat(
                (image_idx[keep, None].float(), boxes[keep], sc[keep, None], cl[keep, None].float()), 1
            ).cpu().numpy()
        out = out[np.argsort(out[:, 0], kind="stable")]  # kelompokkan per gambar, skor tetap urut
        sizes = np.bincount(out[:, 0].astype(np.int64), minlength=len(frames))
        dets = _to_original(out[:, 1:].astype(np.float32), scale, left, top, w, h)
        return [part[:MAX_DETECTIONS] for part in np.split(dets, np.cumsum(sizes)[:-1])]


def build_detector(settings: dict):
    """settings: backend, model, device, gpu_id, cpu_threads, imgsz, rect, conf, iou, batch."""
    common = {k: settings[k] for k in ("gpu_id", "imgsz", "rect", "conf", "iou")}
    if settings["backend"] == "onnx":
        return OnnxDetector(
            Path(settings["model"]), device=settings["device"],
            cpu_threads=settings["cpu_threads"], **common,
        )
    return TorchDetector(Path(settings["model"]), max_batch=settings["batch"], **common)
