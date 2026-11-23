#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCTV weather ML/DL: slice a balanced frame dataset, stratified/temporal train/test
split, train 8 ML/DL classifiers, and evaluate benchmark metrics.

    python -m script_tools.weather_ml slice [--dataset {sp_buahbatu,kordon}]
    python -m script_tools.weather_ml split [--dataset {sp_buahbatu,kordon}]
    python -m script_tools.weather_ml train --model {rf,xgb,lgbm,catboost,mlp,tabnet,ftt,cnn,all} [--dataset {sp_buahbatu,kordon}]
    python -m script_tools.weather_ml evaluate [--dataset {sp_buahbatu,kordon}]

scikit-learn / joblib / torch are imported lazily inside the subcommands that
need them; the deploy image and requirements.txt are untouched (dev-only deps).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import time

import cv2
import numpy as np

from script_tools.weather_vision import (
    sample_clip_frames_indexed,
    ROAD_ROI_REL,
    TEXT_MASK_TOP,
    TEXT_MASK_BOTTOM,
)

try:
    import torch  # noqa: N814
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent


def _to_crop_rel(poly: tuple | list, crop: tuple[float, float, float, float]) -> tuple:
    x0, x1, y0, y1 = crop
    dx = x1 - x0
    dy = y1 - y0
    return tuple(
        (round((x - x0) / dx, 4), round((y - y0) / dy, 4))
        for x, y in poly
    )


@dataclass(frozen=True)
class Dataset:
    key: str                       # CLI value
    camera: str                    # display name (charts/xlsx)
    rain_clip: Path
    sunny_clip: Path
    root: Path                     # parent of dataset_frames*/models_ml/xlsx
    pictures: Path                 # PNG output dir
    n_class: int                   # frames sampled per class
    crop: tuple[float, float, float, float]   # x0, x1, y0, y1 (fractions of full frame)
    roi_rel: tuple                 # road polygon, full-frame fractions (drawn on grid)
    feat_roi: tuple                # polygon used by feature masks, fractions of the STORED jpg
    feat_text_top: float           # text-band mask applied by roi_mask on stored jpg
    feat_text_bottom: float
    split: str                     # "stratified" | "temporal"
    classic: tuple[float, float] | None   # (acc, macroF1) in-sample reference row, or None
    limitations: tuple[str, ...]   # English bullet lines for xlsx section C

    def frames(self, raw: bool = False) -> tuple[Path, Path, Path]:
        d = self.root / ("dataset_frames_raw" if raw else "dataset_frames")
        return d / "all", d / "train", d / "test"

    @property
    def models(self) -> Path:
        return self.root / "models_ml"

    @property
    def xlsx(self) -> Path:
        return self.root / "model_comparison.xlsx"


_KORDON_ROI_REL = (
    (0.02, 0.50),
    (0.72, 0.56),
    (0.90, 0.70),
    (0.90, 0.86),
    (0.02, 0.86),
)
_KORDON_CROP = (0.02, 0.98, 0.08, 0.86)

DATASETS: dict[str, Dataset] = {
    "sp_buahbatu": Dataset(
        key="sp_buahbatu",
        camera="SP Buah Batu",
        rain_clip=BASE_DIR / "weather" / "dataset" / "video" / "sp_buahbatu_rain.mp4",
        sunny_clip=BASE_DIR / "weather" / "dataset" / "video" / "sp_buahbatu_sunny.mp4",
        root=BASE_DIR / "weather",
        pictures=BASE_DIR / "weather" / "output_picture",
        n_class=120,
        crop=(0.02, 0.98, 0.08, 0.92),
        roi_rel=tuple(ROAD_ROI_REL),
        feat_roi=tuple(ROAD_ROI_REL),
        feat_text_top=TEXT_MASK_TOP,
        feat_text_bottom=TEXT_MASK_BOTTOM,
        split="stratified",
        classic=(0.867, 0.865),
        limitations=(
            "• Sunny clip limitation: Sunny frames originate from a single 5-second video clip with near-identical temporal redundancy.",
            "• Generalization caution: High frame-level test accuracy is optimistic due to temporal autocorrelation within the clip.",
            "• Small test set sensitivity: With N=48 test frames, a difference of a single frame corresponds to ~0.021 (2.1%) accuracy.",
            "• Tabular vs CNN domain: Tabular models rely on calibrated road ROI masks; CNN processes raw text-free frame context.",
        ),
    ),
    "kordon": Dataset(
        key="kordon",
        camera="Perempatan Kordon (CCTV-02)",
        rain_clip=BASE_DIR / "weather" / "kordon" / "dataset" / "video" / "cctv_perempatan_kordon_02_20260928T070904Z_rain.mp4",
        sunny_clip=BASE_DIR / "weather" / "kordon" / "dataset" / "video" / "cctv_perempatan_kordon_02_20261004T054311Z_sunny.mp4",
        root=BASE_DIR / "weather" / "kordon",
        pictures=BASE_DIR / "weather" / "kordon" / "output_picture",
        n_class=600,
        crop=_KORDON_CROP,
        roi_rel=_KORDON_ROI_REL,
        feat_roi=_to_crop_rel(_KORDON_ROI_REL, _KORDON_CROP),
        feat_text_top=0.0,
        feat_text_bottom=1.0,
        split="temporal",
        classic=None,
        limitations=(
            "• Single camera, one rain clip (28-09-2026 ~13:50) and one sunny clip (04-10-2026 ~12:25); clip-level generalization to other days is unproven.",
            "• Temporal block split with a 5-frame purge gap limits near-duplicate leakage, but train/test still come from the same two recordings.",
            "• Inner validation splits (CNN/MLP/FTT/TabNet early stopping) are random within the training blocks, so validation curves are optimistic.",
            "• Rain intensity and lighting vary within the rain clip; misclassifications cluster in specific time blocks (see P(rain) heatmap).",
        ),
    ),
}

SIZE = 64              # CNN input size

# Established blue/gray visual template
BLUE = "#2563eb"
GRAY = "#64748b"
MACRO = "#0ea5e9"
FACE = "#f8fafc"
CONFUSE = "#1e3a8a"    # dark blue for confusion title text

MODEL_ORDER = ["rf", "xgb", "lgbm", "catboost", "mlp", "tabnet", "ftt", "cnn", "cnn_paper", "mog"]
MODEL_SPECS = {
    "rf":        ("Random Forest",  "Machine Learning", "scikit-learn",              "24 tabular features"),
    "xgb":       ("XGBoost",        "Machine Learning", "xgboost",                   "24 tabular features"),
    "lgbm":      ("LightGBM",       "Machine Learning", "lightgbm",                  "24 tabular features"),
    "catboost":  ("CatBoost",       "Machine Learning", "catboost",                  "24 tabular features"),
    "mlp":       ("MLP",            "Deep Learning",    "PyTorch",                   "24 tabular features"),
    "tabnet":    ("TabNet",         "Deep Learning",    "pytorch-tabnet",            "24 tabular features"),
    "ftt":       ("FT-Transformer", "Deep Learning",    "PyTorch",                   "24 tabular features"),
    "cnn":       ("CNN",            "Deep Learning",    "PyTorch",                   "64x64 RGB image"),
    "cnn_paper": ("CNN_paper",      "Deep Learning",    "PyTorch",                   "150x150 RGB image"),
    "mog":       ("MOG",            "Machine Learning", "OpenCV MOG2 + scikit-learn", "6 MOG2 motion features"),
}


def _ds_paths(ds: Dataset, raw: bool) -> tuple[Path, Path, Path]:
    """Return (all, train, test) dirs; raw skips the text-free crop."""
    return ds.frames(raw)


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require_exists(p: Path) -> None:
    if not p.is_file():
        _abort(f"missing input {p.name} (run `slice` first)")


# --------------------------------------------------------------------------- #
# Step 1 — slice balanced frame dataset
# --------------------------------------------------------------------------- #

def _crop(frame: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = crop
    rx0, rx1 = int(x0 * w), int(x1 * w)
    ry0, ry1 = int(y0 * h), int(y1 * h)
    return frame[ry0:ry1, rx0:rx1]


def _mask_text_bands(frame: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    """Black out the timestamp/date bands based on crop boundaries."""
    h = frame.shape[0]
    out = frame.copy()
    y0, y1 = crop[2], crop[3]
    out[:int(y0 * h), :] = 0
    out[int(y1 * h):, :] = 0
    return out

def _clear_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.jpg"):
        f.unlink()


def _draw_roi(img: np.ndarray, ds: Dataset) -> np.ndarray:
    """Overlay the road ROI polygon (blue) on a cropped frame, in crop coords."""
    ih, iw = img.shape[:2]
    pts = []
    crop_rel = _to_crop_rel(ds.roi_rel, ds.crop)
    for cx_rel, cy_rel in crop_rel:
        cx = int(cx_rel * iw)
        cy = int(cy_rel * ih)
        pts.append((cx, cy))
    cv2.polylines(img, [np.array(pts, np.int32)], True, (37, 99, 235), 2)
    return img

def cmd_slice(args: argparse.Namespace) -> int:
    raw = getattr(args, "raw", False)
    ds = getattr(args, "ds", DATASETS["sp_buahbatu"])
    all_dir, _, _ = ds.frames(raw)
    clips = {
        "rain": ds.rain_clip,
        "sunny": ds.sunny_clip,
    }
    counts = {}
    saved = {}
    saved_idx = {}
    for cls in ("rain", "sunny"):
        p = clips[cls]
        _require_exists(p)
        saved[cls], saved_idx[cls] = sample_clip_frames_indexed(p, ds.n_class)
        counts[cls] = len(saved[cls])
        if counts[cls] == 0:
            _abort(f"no decodable frames from {p.name}")
        if raw:
            saved[cls] = [_mask_text_bands(f, ds.crop) for f in saved[cls]]
        else:
            saved[cls] = [_crop(f, ds.crop) for f in saved[cls]]

    target = min(counts.values())
    manifest: dict[str, dict[str, int]] = {"rain": {}, "sunny": {}}
    for cls in ("rain", "sunny"):
        frames = saved[cls]
        f_idxs = saved_idx[cls]
        if len(frames) != target:
            idxs = np.linspace(0, len(frames) - 1, target, dtype=np.int64)
            frames = [frames[i] for i in idxs]
            f_idxs = [f_idxs[i] for i in idxs]
        out = all_dir / cls
        _clear_dir(out)
        for i, fr in enumerate(frames):
            fname = f"frame_{i:03d}.jpg"
            cv2.imwrite(
                str(out / fname), fr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            manifest[cls][fname] = int(f_idxs[i])
        counts[cls] = len(frames)

    manifest_path = all_dir.parent / "frame_index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _write_grid(counts, ds=ds, raw=raw)
    print(f"rain={counts['rain']} sunny={counts['sunny']}")
    print(f"saved to {all_dir}")
    print(f"saved manifest to {manifest_path}")
    return 0


def _write_grid(counts: dict, ds: Dataset, raw: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dir, _, _ = ds.frames(raw)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), facecolor=FACE)
    row_lbl = {"rain": BLUE, "sunny": GRAY}
    for r, cls in enumerate(("rain", "sunny")):
        files = sorted((all_dir / cls).glob("*.jpg"))
        step = max(1, len(files) // 3)
        for c in range(3):
            pf = files[min(c * step, len(files) - 1)]
            ax = axes[r, c]
            img = cv2.imread(str(pf))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not raw:
                _draw_roi(img, ds)
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"{cls.capitalize()} — Frame {pf.stem[len('frame_'):]}",
                             fontsize=9)
            ax.axis("off")
    fig.tight_layout(rect=(0.045, 0, 1, 1))
    for r, cls in enumerate(("rain", "sunny")):
        yc = axes[r, 0].get_position().y0 + axes[r, 0].get_position().height / 2
        fig.text(0.012, yc, cls.upper(), rotation=90, ha="left", va="center",
                 fontsize=12, fontweight="bold", color=row_lbl[cls])
    ds.pictures.mkdir(parents=True, exist_ok=True)
    fig.savefig(ds.pictures / ("_ml_slice_grid_raw.png" if raw else "_ml_slice_grid.png"), dpi=110)
    plt.close(fig)



# --------------------------------------------------------------------------- #
# Step 2 — stratified train/test split (file moves)
# --------------------------------------------------------------------------- #

def _split_one(files: list, test_size: float) -> tuple:
    from sklearn.model_selection import train_test_split
    labels = [0] * len(files)
    tr, te = train_test_split(files, test_size=test_size,
                              random_state=42, stratify=labels)
    return tr, te


def _split_temporal(
    files: list,
    n_blocks: int = 10,
    test_blocks: tuple = (3, 8),
    purge: int = 5,
) -> tuple[list, list, list]:
    n = len(files)
    if n < n_blocks * 5:
        _abort("temporal split needs >= 50 frames per class")
    blocks = np.array_split(np.arange(n), n_blocks)
    test_idx_set = set()
    test_ranges = []
    for b_idx in test_blocks:
        blk = blocks[b_idx]
        test_idx_set.update(blk.tolist())
        test_ranges.append((int(blk[0]), int(blk[-1])))

    purged_idx_set = set()
    for start, end in test_ranges:
        # [start - purge, start)
        for i in range(max(0, start - purge), start):
            if i not in test_idx_set:
                purged_idx_set.add(i)
        # (end, end + purge]
        for i in range(end + 1, min(n, end + purge + 1)):
            if i not in test_idx_set:
                purged_idx_set.add(i)

    train_files = [files[i] for i in range(n) if i not in test_idx_set and i not in purged_idx_set]
    test_files = [files[i] for i in sorted(test_idx_set)]
    purged_files = [files[i] for i in sorted(purged_idx_set)]
    return train_files, test_files, purged_files


def cmd_split(args: argparse.Namespace) -> int:
    raw = getattr(args, "raw", False)
    ds = getattr(args, "ds", DATASETS["sp_buahbatu"])
    all_dir, train_dir, test_dir = ds.frames(raw)
    purged_dir = all_dir.parent / "purged"

    for split_dir, cls in ((train_dir, "rain"), (train_dir, "sunny"),
                           (test_dir, "rain"), (test_dir, "sunny")):
        _clear_dir(split_dir / cls)
    if ds.split == "temporal":
        for cls in ("rain", "sunny"):
            _clear_dir(purged_dir / cls)

    rain_all = sorted((all_dir / "rain").glob("*.jpg"))
    sunny_all = sorted((all_dir / "sunny").glob("*.jpg"))
    if not rain_all or not sunny_all:
        _abort("slice first (empty per-class dataset)")

    if ds.split == "temporal":
        rain_tr, rain_te, rain_pg = _split_temporal(rain_all)
        sunny_tr, sunny_te, sunny_pg = _split_temporal(sunny_all)
        moves = [
            (rain_tr, train_dir / "rain"), (sunny_tr, train_dir / "sunny"),
            (rain_te, test_dir / "rain"), (sunny_te, test_dir / "sunny"),
            (rain_pg, purged_dir / "rain"), (sunny_pg, purged_dir / "sunny"),
        ]
    else:
        rain_tr, rain_te = _split_one(rain_all, 0.2)
        sunny_tr, sunny_te = _split_one(sunny_all, 0.2)
        moves = [
            (rain_tr, train_dir / "rain"), (sunny_tr, train_dir / "sunny"),
            (rain_te, test_dir / "rain"), (sunny_te, test_dir / "sunny"),
        ]

    for files, dest in moves:
        for f in files:
            shutil.move(str(f), str(dest / f.name))

    cells = {
        (train_dir / "rain"): len(rain_tr), (train_dir / "sunny"): len(sunny_tr),
        (test_dir / "rain"): len(rain_te), (test_dir / "sunny"): len(sunny_te),
    }
    if any(c == 0 for c in cells.values()):
        _abort("empty split cell; cannot train on missing class")

    if ds.split == "temporal":
        print(f"train: rain={len(rain_tr)} sunny={len(sunny_tr)}"
              f" | test: rain={len(rain_te)} sunny={len(sunny_te)}"
              f" | purged: rain={len(rain_pg)} sunny={len(sunny_pg)}")
    else:
        print(f"train: rain={cells[train_dir/'rain']} sunny={cells[train_dir/'sunny']}"
              f" | test: rain={cells[test_dir/'rain']} sunny={cells[test_dir/'sunny']}")
    _write_split_dist(ds=ds, raw=raw)
    return 0


def _write_split_dist(ds: Dataset, raw: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dir, train_dir, test_dir = ds.frames(raw)
    purged_dir = all_dir.parent / "purged"
    rain_tr = len(list((train_dir / "rain").glob("*.jpg")))
    sunny_tr = len(list((train_dir / "sunny").glob("*.jpg")))
    rain_te = len(list((test_dir / "rain").glob("*.jpg")))
    sunny_te = len(list((test_dir / "sunny").glob("*.jpg")))
    has_purged = purged_dir.exists() and any(purged_dir.glob("*/*.jpg"))

    x = np.arange(2)
    tr = [rain_tr, sunny_tr]
    te = [rain_te, sunny_te]
    all_vals = tr + te

    fig, ax = plt.subplots(figsize=(5.8 if has_purged else 5.2, 3.8), facecolor=FACE)
    if has_purged:
        rain_pg = len(list((purged_dir / "rain").glob("*.jpg")))
        sunny_pg = len(list((purged_dir / "sunny").glob("*.jpg")))
        pg = [rain_pg, sunny_pg]
        all_vals += pg
        w = 0.26
        bars1 = ax.bar(x - w, tr, width=w, label="Train", color=BLUE)
        bars2 = ax.bar(x, te, width=w, label="Test", color=GRAY)
        bars3 = ax.bar(x + w, pg, width=w, label="Purged (gap)", color="#cbd5e1")
        bar_groups = (bars1, bars2, bars3)
    else:
        bars1 = ax.bar(x - 0.19, tr, width=0.38, label="Train", color=BLUE)
        bars2 = ax.bar(x + 0.19, te, width=0.38, label="Test", color=GRAY)
        bar_groups = (bars1, bars2)

    for bars in bar_groups:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{int(b.get_height())}", ha="center", va="bottom",
                    fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Rain", "Sunny"])
    ax.set_ylabel("Frames")
    ax.set_ylim(0, max(all_vals) * 1.22)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Train / test split — {ds.split} ({ds.camera})"
                 + (" — raw" if raw else ""))
    fig.tight_layout()
    ds.pictures.mkdir(parents=True, exist_ok=True)
    fig.savefig(ds.pictures / ("_ml_split_dist_raw.png" if raw else "_ml_split_dist.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Shared metrics + plots
# --------------------------------------------------------------------------- #

def _compute_metrics(y_true, y_pred, labels=("rain", "sunny")) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
    )
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    per = {}
    for i, lbl in enumerate(labels):
        per[lbl] = {"precision": float(pr[i]), "recall": float(rc[i]),
                    "f1": float(f1[i])}
    macro = {"precision": float(np.mean(pr)), "recall": float(np.mean(rc)),
             "f1": float(np.mean(f1))}
    # positive = rain (row/col 0):
    return {"accuracy": acc, "per_class": per, "macro": macro,
            "confusion": cm.tolist()}


def _write_training_env(ds: Dataset) -> None:
    import importlib.metadata
    import platform
    import sklearn
    lines = [f"python={platform.python_version()}",
             f"sklearn={sklearn.__version__}"]
    try:
        import torch
        lines.append(f"torch={torch.__version__}")
    except Exception:
        lines.append("torch=missing")
    lines.append(f"numpy={np.__version__}")
    for dist in ("xgboost", "lightgbm", "catboost", "pytorch-tabnet", "xlsxwriter"):
        try:
            v = importlib.metadata.version(dist)
            lines.append(f"{dist}={v}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{dist}=missing")
    ds.models.mkdir(parents=True, exist_ok=True)
    (ds.models / "TRAINING_ENV.txt").write_text("\n".join(lines) + "\n")


def _build_metrics_record(
    key: str,
    y_true,
    y_pred,
    p_rain,
    *,
    ds: Dataset,
    n_train: int,
    train_seconds: float,
    infer_seconds: float,
    params: dict,
    feature_importance: dict | None = None,
    history: dict | None = None,
    test_names: list[str] | None = None,
    file_tag: str = "",
) -> dict:
    from sklearn.metrics import roc_auc_score

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    p_rain_arr = np.asarray(p_rain, dtype=np.float64)

    # 0 = rain, 1 = sunny
    y_true_lbl = np.where(y_true_arr == 0, "rain", "sunny")
    y_pred_lbl = np.where(y_pred_arr == 0, "rain", "sunny")

    base_metrics = _compute_metrics(y_true_lbl, y_pred_lbl, labels=("rain", "sunny"))

    y_true_rain_binary = (y_true_arr == 0).astype(int)
    try:
        auc_val = float(roc_auc_score(y_true_rain_binary, p_rain_arr))
    except Exception:
        auc_val = 0.5

    spec = MODEL_SPECS.get(key, (key.upper(), "Unknown", "custom", "features"))
    n_test = len(y_true_arr)
    infer_ms = (float(infer_seconds) * 1000.0 / n_test) if n_test > 0 else 0.0

    record = {
        **base_metrics,
        "model": key,
        "display_name": spec[0],
        "family": spec[1],
        "library": spec[2],
        "input": spec[3],
        "roc_auc": round(auc_val, 4),
        "train_seconds": round(float(train_seconds), 4),
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "params": params,
        "feature_importance": feature_importance,
        "history": history,
        "y_true": [int(y) for y in y_true_arr],
        "p_rain": [round(float(p), 6) for p in p_rain_arr],
        "test_names": test_names if test_names is not None else [],
    }

    ds.models.mkdir(parents=True, exist_ok=True)
    out_path = ds.models / f"metrics_{key}{file_tag}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _load_images(split_dir: Path, size: int = SIZE) -> tuple:
    xs, ys, names = [], [], []
    for cls, label in (("rain", 0), ("sunny", 1)):
        for p in sorted((split_dir / cls).glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            xs.append(img.astype(np.float32) / 255.0)
            ys.append(label)
            names.append(f"{cls}/{p.name}")
    if not xs:
        _abort("no images (run slice + split first)")
    return np.stack(xs), np.array(ys), names

# _load_rf_features moved to script_tools.weather_tabular.load_tabular


def _write_misclass_cm(cm_path: Path, cm, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=FACE)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rain", "Sunny"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Rain", "Sunny"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black",
                    fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=110)
    plt.close(fig)


def _write_metrics_bars(path: Path, metrics: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Rain", "Sunny", "Macro"]
    colors = [BLUE, GRAY, MACRO]
    metrics_keys = ["precision", "recall", "f1"]
    x = np.arange(len(metrics_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=FACE)
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if lbl == "Macro":
            vals = [metrics["macro"][key] for key in metrics_keys]
        else:
            vals = [metrics["per_class"][lbl.lower()][key]
                    for key in metrics_keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    acc = metrics["accuracy"]
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.46)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.legend(loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0))
    ax.text(0.03, 0.935, f"Test accuracy {acc:.3f}", transform=ax.transAxes,
            fontsize=9, color="crimson",
            bbox=dict(boxstyle="round,pad=0.35", fc="none", ec="crimson"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Step 3 — RandomForest
# --------------------------------------------------------------------------- #

# cmd_train_rf replaced by script_tools.weather_tabular.train_tabular("rf")


def _write_featimp(path: Path, importances: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importances.keys())
    vals = [importances[n] for n in names]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.5, max(4.0, len(names) * 0.22)), facecolor=FACE)
    ax.barh(np.array(names)[order], np.array(vals)[order], color=BLUE)
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)



def _require_torch():
    if not _HAS_TORCH:
        _abort("torch not installed (pip install torch --index-url "
               "https://download.pytorch.org/whl/cpu) to train the CNN")


if _HAS_TORCH:

    class _CNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Dropout(0.25),
                torch.nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    class _PaperCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Hakim & Prawito 2020 Table 1 Training-5:
            # Conv(32, 3x3) -> MaxPool(2x2) -> Conv(64, 2x2) -> MaxPool(2x2) -> FC -> Softmax
            # Input 150x150: 150 -> 148 -> 74 -> 73 -> 36. 64 * 36 * 36 = 82944.
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 2), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 36 * 36, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    class _ImgDS(torch.utils.data.Dataset):
        def __init__(self, x, y, aug=False):
            self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
            self.y = torch.tensor(y, dtype=torch.long)
            self.aug = aug

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x, y = self.x[i], self.y[i]
            if self.aug and torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            return x, y


def _train_image_cnn(
    args: argparse.Namespace,
    *,
    key: str,
    model: "torch.nn.Module",
    size: int,
    normalize: bool,
    aug: bool,
    epochs: int,
    batch_size: int,
    params: dict,
) -> int:
    import torch
    from sklearn.model_selection import train_test_split

    ds = getattr(args, "ds", DATASETS["sp_buahbatu"])
    raw = getattr(args, "raw", False)
    _, train_dir, test_dir = ds.frames(raw)
    tag = "_raw" if raw else ""
    disp_name = MODEL_SPECS[key][0]
    title = f"{disp_name} (raw frames) — test metrics ({ds.camera})" if raw else f"{disp_name} — test metrics ({ds.camera})"

    torch.manual_seed(42)
    ds.models.mkdir(parents=True, exist_ok=True)
    ds.pictures.mkdir(parents=True, exist_ok=True)
    _write_training_env(ds)

    Xtr, ytr, _ = _load_images(train_dir, size=size)
    Xte, yte, test_names = _load_images(test_dir, size=size)

    if normalize:
        mean = Xtr.reshape(-1, 3).mean(0)
        std = Xtr.reshape(-1, 3).std(0) + 1e-6
        (ds.models / f"{key}_norm{tag}.json").write_text(
            json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2) + "\n")

        def _norm(x):
            return (x - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)

        Xtr_n, Xte_n = _norm(Xtr), _norm(Xte)
    else:
        Xtr_n, Xte_n = Xtr, Xte

    fit_idx, val_idx = train_test_split(
        np.arange(len(ytr)), test_size=0.15, random_state=42,
        stratify=ytr)
    fit_ds = _ImgDS(Xtr_n[fit_idx], ytr[fit_idx], aug=aug)
    val_ds = _ImgDS(Xtr_n[val_idx], ytr[val_idx], aug=False)
    test_ds = _ImgDS(Xte_n, yte, aug=False)

    loader = torch.utils.data.DataLoader
    fit_ld = loader(fit_ds, batch_size=batch_size, shuffle=True)
    val_ld = loader(val_ds, batch_size=64, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 1e-4))

    best_loss = float("inf")
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    ckpt_path = ds.models / f"{key}_weather{tag}.pt"

    train_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in fit_ld:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
        tl /= len(fit_ds)

        model.eval()
        vl = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                out = model(xb)
                vl += loss_fn(out, yb).item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
        vl /= len(val_ds)
        vacc = correct / total
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        print(f"epoch {ep:02d} | train_loss {tl:.4f} | val_loss {vl:.4f} "
              f"| val_acc {vacc:.3f}")
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), str(ckpt_path))
    train_seconds = time.perf_counter() - train_start

    model.load_state_dict(torch.load(str(ckpt_path)))
    model.eval()
    yp = []
    p_rain = []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for xb, _ in test_ds:
            logits = model(xb.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            p_rain.append(float(probs[0]))
            yp.append(int(logits.argmax(1).item()))
    infer_seconds = time.perf_counter() - infer_start

    yp = np.array(yp)
    p_rain = np.array(p_rain)

    record = _build_metrics_record(
        key,
        yte,
        yp,
        p_rain,
        ds=ds,
        n_train=len(ytr),
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
        params=params,
        history=hist,
        test_names=test_names,
        file_tag=tag,
    )

    _write_curves(hist, ds.pictures / f"_curves_{key}{tag}.png", title=title)
    _write_metrics_bars(ds.pictures / f"_metrics_{key}{tag}.png", record,
                        title)
    _write_misclass_cm(ds.pictures / f"_cm_{key}{tag}.png",
                       record["confusion"], title.replace("metrics", "matrix"))

    print(f"{disp_name} test accuracy={record['accuracy']:.3f} "
          f"macro-F1={record['macro']['f1']:.3f} "
          f"AUC={record['roc_auc']:.3f}")
    return 0


def cmd_train_cnn(args: argparse.Namespace) -> int:
    _require_torch()
    epochs = 40 if getattr(args, "fallback", False) else 20
    params = {
        "arch": "Conv16-Conv32-Conv64-GAP-Dropout0.25-Linear2",
        "input": "64x64 RGB",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": epochs,
    }
    return _train_image_cnn(
        args,
        key="cnn",
        model=_CNN(),
        size=64,
        normalize=True,
        aug=True,
        epochs=epochs,
        batch_size=16,
        params=params,
    )


def cmd_train_cnn_paper(args: argparse.Namespace) -> int:
    _require_torch()
    params = {
        "source": "Hakim & Prawito 2020, J. Phys.: Conf. Ser. 1528 012010, Table 1 Training-5",
        "arch": "Conv32(3x3)-ReLU-MaxPool2-Conv64(2x2)-ReLU-MaxPool2-FC(82944->2)-Softmax",
        "input": "150x150 RGB /255",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 50,
    }
    return _train_image_cnn(
        args,
        key="cnn_paper",
        model=_PaperCNN(),
        size=150,
        normalize=False,
        aug=False,
        epochs=50,
        batch_size=32,
        params=params,
    )


def _write_curves(hist: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), facecolor=FACE)
    ax1 = axes[0]
    ax1.plot(epochs, hist["train_loss"], color=BLUE, label="train loss")
    ax1.plot(epochs, hist["val_loss"], color=GRAY, label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2 = axes[1]
    ax2.plot(epochs, hist["val_acc"], color=MACRO, label="val accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Step 5 — evaluation comparison
# --------------------------------------------------------------------------- #

def _load_metrics(name: str, ds: Dataset) -> dict | None:
    p = ds.models / name
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def cmd_evaluate(args: argparse.Namespace) -> int:
    from script_tools.weather_excel import write_workbook

    ds = getattr(args, "ds", DATASETS["sp_buahbatu"])
    ds.models.mkdir(parents=True, exist_ok=True)
    ds.pictures.mkdir(parents=True, exist_ok=True)

    records = []
    for key in MODEL_ORDER:
        rec = _load_metrics(f"metrics_{key}.json", ds)
        if rec is None or "display_name" not in rec:
            print(f"{key} metrics missing/outdated — training now…")
            _train_one(key, args)
            rec = _load_metrics(f"metrics_{key}.json", ds)
        records.append(rec)

    hdr = f"{'model':<18}{'family':<18}{'accuracy':>10}{'macroF1':>10}{'F1rain':>10}{'F1sunny':>10}{'AUC':>10}{'train_s':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))

    if ds.classic is not None:
        c_acc, c_f1 = ds.classic
        print(f"{'Classical*':<18}{'Heuristic/CV':<18}{c_acc:>10.3f}{c_f1:>10.3f}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")

    for r in records:
        m_name = r.get("display_name", r.get("model", ""))
        fam = r.get("family", "")
        acc = r.get("accuracy", 0.0)
        mf1 = r.get("macro", {}).get("f1", 0.0)
        f1r = r.get("per_class", {}).get("rain", {}).get("f1", 0.0)
        f1s = r.get("per_class", {}).get("sunny", {}).get("f1", 0.0)
        auc = r.get("roc_auc", 0.0)
        tr_s = r.get("train_seconds", 0.0)
        print(f"{m_name:<18}{fam:<18}{acc:>10.3f}{mf1:>10.3f}{f1r:>10.3f}{f1s:>10.3f}{auc:>10.3f}{tr_s:>10.2f}")

    _write_compare(records, ds)
    _write_roc_all(records, ds)
    _write_speed(records, ds)
    _write_prain_heatmap(records, ds)

    write_workbook(records, ds.xlsx, ds)
    print(f"\nExcel benchmark report exported to: {ds.xlsx}")

    rf_rec = next((r for r in records if r.get("model") == "rf"), {})
    cnn_rec = next((r for r in records if r.get("model") == "cnn"), {})
    ok = rf_rec.get("accuracy", 0.0) >= 0.85 and cnn_rec.get("accuracy", 0.0) >= 0.80
    print("SUCCESS" if ok else "THRESHOLDS NOT MET (rf>=0.85, cnn>=0.80)")
    return 0


def _write_compare(records: list[dict], ds: Dataset) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    accs = []
    f1s = []
    if ds.classic is not None:
        names.append("Classical*")
        accs.append(ds.classic[0])
        f1s.append(ds.classic[1])

    for r in records:
        names.append(r.get("display_name", r.get("model", "")))
        accs.append(r.get("accuracy", 0.0))
        f1s.append(r.get("macro", {}).get("f1", 0.0))

    n = len(names)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11.0, 4.6), facecolor=FACE)
    b1 = ax.bar(x - 0.19, accs, width=0.38, label="Accuracy", color=BLUE)
    b2 = ax.bar(x + 0.19, f1s, width=0.38, label="Macro F1", color=GRAY)

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    suffix = "; *in-sample reference)" if ds.classic is not None else ")"
    fig.suptitle(f"Model comparison — {ds.camera} (test split{suffix}")
    fig.tight_layout()
    ds.pictures.mkdir(parents=True, exist_ok=True)
    fig.savefig(ds.pictures / "_compare_models.png", dpi=110)
    plt.close(fig)


def _write_roc_all(records: list[dict], ds: Dataset) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    for r in records:
        if "p_rain" not in r or not r["p_rain"]:
            print("skip ROC/heatmap: metrics lack p_rain (retrain with this version)")
            return

    colors = [
        "#2563eb", "#0ea5e9", "#10b981", "#84cc16",
        "#f59e0b", "#ec4899", "#8b5cf6", "#64748b",
        "#ef4444", "#14b8a6",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor=FACE)
    for i, r in enumerate(records):
        y_true = np.asarray(r["y_true"])
        p_rain = np.asarray(r["p_rain"])
        y_true_rain = (y_true == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_rain, p_rain)
        col = colors[i % len(colors)]
        auc_val = r.get("roc_auc", 0.5)
        disp = r.get("display_name", r.get("model", ""))
        ax.plot(fpr, tpr, color=col, lw=2.0, label=f"{disp} (AUC {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate (Sunny misclassified as Rain)")
    ax.set_ylabel("True Positive Rate (Rain detected)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"ROC Curves — {ds.camera} (Rain Detection)")
    fig.tight_layout()
    fig.savefig(ds.pictures / "_roc_all.png", dpi=110)
    plt.close(fig)


def _write_speed(records: list[dict], ds: Dataset) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.get("display_name", r.get("model", "")) for r in records]
    tr_times = [r.get("train_seconds", 0.0) for r in records]
    inf_times = [r.get("infer_ms_per_sample", 0.0) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=FACE)
    y = np.arange(len(names))

    b1 = ax1.barh(y, tr_times, color=BLUE, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Training Time (seconds)")
    ax1.set_title("Training Duration", fontsize=10)
    ax1.invert_yaxis()
    ax1.spines[["top", "right"]].set_visible(False)
    for b in b1:
        ax1.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.2f}s", va="center", fontsize=8)

    b2 = ax2.barh(y, inf_times, color=MACRO, height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Inference Latency (ms / frame)")
    ax2.set_title("Inference Latency", fontsize=10)
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)
    for b in b2:
        ax2.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.2f}ms", va="center", fontsize=8)

    fig.suptitle(f"Model Training Time & Latency — {ds.camera}")
    fig.tight_layout()
    fig.savefig(ds.pictures / "_speed_models.png", dpi=110)
    plt.close(fig)


def _write_prain_heatmap(records: list[dict], ds: Dataset) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in records:
        if "p_rain" not in r or not r.get("test_names"):
            return

    sample_names = records[0]["test_names"]
    def _key_fn(item):
        cls, fname = item.split("/")
        num = int("".join(filter(str.isdigit, fname)))
        return (0 if cls == "rain" else 1, num)

    sorted_indices = sorted(range(len(sample_names)), key=lambda idx: _key_fn(sample_names[idx]))
    sorted_names = [sample_names[i] for i in sorted_indices]

    model_names = [r.get("display_name", r.get("model", "")) for r in records]
    matrix = []
    for r in records:
        row = [r["p_rain"][i] for i in sorted_indices]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12.0, 4.6), facecolor=FACE)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)

    rain_count = sum(1 for n in sorted_names if n.startswith("rain/"))
    if 0 < rain_count < len(sorted_names):
        ax.axvline(rain_count - 0.5, color="black", lw=2.0)

    for i in range(1, len(sorted_names)):
        cls_curr, fn_curr = sorted_names[i].split("/")
        cls_prev, fn_prev = sorted_names[i - 1].split("/")
        if cls_curr == cls_prev:
            n_curr = int("".join(filter(str.isdigit, fn_curr)))
            n_prev = int("".join(filter(str.isdigit, fn_prev)))
            if n_curr - n_prev > 1:
                ax.axvline(i - 0.5, color="gray", lw=1.0, linestyle="--", alpha=0.7)

    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Held-out Test Frames (Time Order: Rain blocks [left] | Sunny blocks [right])")
    ax.set_title(f"P(rain) per test frame — {ds.camera} (left: rain, right: sunny, time order)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("P(rain)")
    fig.tight_layout()
    fig.savefig(ds.pictures / "_prain_heatmap.png", dpi=110)
    plt.close(fig)
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="sp_buahbatu",
        help="dataset profile (default: sp_buahbatu)",
    )

    parser = argparse.ArgumentParser(
        prog="python -m script_tools.weather_ml")
    sub = parser.add_subparsers(dest="cmd")

    p_slice = sub.add_parser("slice", parents=[common], help="extract balanced frame dataset")
    p_slice.add_argument("--raw", action="store_true",
                         help="skip the text-free crop (raw 640x480 frames)")
    p_slice.set_defaults(fn=cmd_slice)

    p_split = sub.add_parser("split", parents=[common], help="stratified/temporal train/test move")
    p_split.add_argument("--raw", action="store_true",
                         help="use the raw dataset")
    p_split.set_defaults(fn=cmd_split)

    p_train = sub.add_parser("train", parents=[common], help="train a model")
    p_train.add_argument("--model", choices=MODEL_ORDER + ["all"], required=True)
    p_train.add_argument("--raw", action="store_true",
                         help="use the raw dataset")
    p_train.add_argument("--fallback", action="store_true",
                         help="CNN: retrain 40 epochs + rotation aug")
    p_train.set_defaults(fn=cmd_train)

    p_eval = sub.add_parser("evaluate", parents=[common], help="compare all models")
    p_eval.set_defaults(fn=cmd_evaluate)

    args = parser.parse_args(argv)
    if not hasattr(args, "fn"):
        parser.print_help()
        return 2
    args.ds = DATASETS[args.dataset]
    return int(args.fn(args))


def _train_one(key: str, args: argparse.Namespace) -> None:
    if key == "cnn":
        cmd_train_cnn(args)
    elif key == "cnn_paper":
        cmd_train_cnn_paper(args)
    elif key == "mog":
        if getattr(args, "raw", False):
            _abort("mog does not support --raw")
        from script_tools.weather_mog import train_mog
        train_mog(args.ds)
    else:
        from script_tools.weather_tabular import train_tabular
        train_tabular(key, args.ds)


def cmd_train(args: argparse.Namespace) -> int:
    if args.model == "all":
        for m in MODEL_ORDER:
            print(f"\n=== Training {m.upper()} ===")
            _train_one(m, args)
        return 0
    _train_one(args.model, args)
    return 0

if __name__ == "__main__":
    sys.exit(main())