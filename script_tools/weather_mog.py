"""
MOG motion-detection classifier for pelindung weather benchmark.
Uses OpenCV MOG2 background subtraction on video windows preceding sampled frames,
computes motion mask morphology features, and classifies with Logistic Regression.
"""

from collections import deque
import json
from pathlib import Path
import sys
import time

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from script_tools.weather_vision import (
    WORK_SIZE,
    roi_mask,
)
from script_tools.weather_ml import (
    Dataset,
    DATASETS,
    _crop,
    _abort,
    _require_exists,
    _write_training_env,
    _build_metrics_record,
    _write_metrics_bars,
    _write_misclass_cm,
    _write_featimp,
    BLUE,
    GRAY,
    FACE,
)

MOG_FEATURES = [
    "fg_ratio_roi",
    "fg_ratio_full",
    "small_blob_density",
    "large_blob_density",
    "small_fg_share",
    "fg_ratio_roi_std",
]

MOG_HISTORY = 100
MOG_VAR_THRESHOLD = 16
MOG_WINDOW = 5
SMALL_BLOB_MIN = 2
SMALL_BLOB_MAX = 40  # pixels at WORK_SIZE (320x240)


def _mask_features(masks: list[np.ndarray], roi: np.ndarray) -> list[float]:
    """Compute 6 summary features across the window of boolean foreground masks."""
    fg_rois = []
    fg_fulls = []
    small_densities = []
    large_densities = []
    small_shares = []

    roi_count = max(1, int(roi.sum()))

    for m in masks:
        m_roi = m & roi
        fg_roi = float(m_roi.sum()) / roi_count
        fg_full = float(m.mean())
        fg_rois.append(fg_roi)
        fg_fulls.append(fg_full)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            m_roi.astype(np.uint8), connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
        small_mask = (areas >= SMALL_BLOB_MIN) & (areas <= SMALL_BLOB_MAX)
        large_mask = areas > SMALL_BLOB_MAX

        small_count = int(small_mask.sum())
        large_count = int(large_mask.sum())
        small_area_sum = float(areas[small_mask].sum()) if small_count > 0 else 0.0

        small_density = (small_count / roi_count) * 10000.0
        large_density = (large_count / roi_count) * 10000.0
        m_roi_sum = max(1, int(m_roi.sum()))
        small_share = small_area_sum / m_roi_sum

        small_densities.append(small_density)
        large_densities.append(large_density)
        small_shares.append(small_share)

    return [
        float(np.mean(fg_rois)),
        float(np.mean(fg_fulls)),
        float(np.mean(small_densities)),
        float(np.mean(large_densities)),
        float(np.mean(small_shares)),
        float(np.std(fg_rois)),
    ]


def extract_mog_features(
    ds: Dataset, keep: dict[str, str] | None = None
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, np.ndarray]], float]:
    """
    Sequentially decode video clips for rain and sunny, running MOG2.
    Extract features for frames present in frame_index.json.
    `keep` specifies {cls: name} to preserve example foreground masks.
    """
    manifest_path = ds.frames()[0].parent / "frame_index.json"
    if not manifest_path.is_file():
        _abort(f"missing {manifest_path} (re-run `slice --dataset {ds.key}`)")

    manifest: dict[str, dict[str, int]] = json.loads(manifest_path.read_text())

    h, w = WORK_SIZE[1], WORK_SIZE[0]
    roi = roi_mask(
        h,
        w,
        roi_rel=ds.feat_roi,
        text_top=ds.feat_text_top,
        text_bottom=ds.feat_text_bottom,
    )

    features: dict[str, dict[str, list[float]]] = {"rain": {}, "sunny": {}}
    example_masks: dict[str, dict[str, np.ndarray]] = {"rain": {}, "sunny": {}}

    clips = {"rain": ds.rain_clip, "sunny": ds.sunny_clip}

    t0 = time.perf_counter()

    for cls in ("rain", "sunny"):
        clip_path = clips[cls]
        _require_exists(clip_path)

        targets_map = manifest.get(cls, {})
        # Map target decoded frame index -> list of file names (usually 1)
        idx_to_names: dict[int, list[str]] = {}
        for fname, idx in targets_map.items():
            idx_to_names.setdefault(int(idx), []).append(fname)

        if not idx_to_names:
            continue

        max_target = max(idx_to_names.keys())

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            _abort(f"cannot open video {clip_path}")

        sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG_HISTORY, varThreshold=MOG_VAR_THRESHOLD, detectShadows=False
        )
        window_buf: deque[np.ndarray] = deque(maxlen=MOG_WINDOW)

        seen = 0
        keep_name = keep.get(cls) if keep else None

        while seen <= max_target:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cropped = _crop(frame, ds.crop)
            small = cv2.resize(cropped, WORK_SIZE, interpolation=cv2.INTER_AREA)
            fg_mask = (sub.apply(small) > 127)
            window_buf.append(fg_mask)

            if seen in idx_to_names:
                fnames = idx_to_names[seen]
                feat = _mask_features(list(window_buf), roi)
                for fn in fnames:
                    features[cls][fn] = feat
                    if keep_name and fn == keep_name:
                        example_masks[cls][fn] = fg_mask.copy()

            seen += 1

        cap.release()

        # Check if all targets were decoded
        for fname, idx in targets_map.items():
            if fname not in features[cls]:
                _abort(f"{clip_path.name}: video ended before target frame index {idx} ({fname})")

    elapsed = time.perf_counter() - t0
    return features, example_masks, elapsed


def _write_mog_examples(
    ds: Dataset, examples: dict[str, dict[str, np.ndarray]], keep: dict[str, str]
) -> None:
    """Save 2x2 figure showing rain & sunny crops vs MOG2 masks with ROI outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = WORK_SIZE[1], WORK_SIZE[0]
    roi = roi_mask(
        h,
        w,
        roi_rel=ds.feat_roi,
        text_top=ds.feat_text_top,
        text_bottom=ds.feat_text_bottom,
    )
    contours, _ = cv2.findContours(
        roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), facecolor=FACE)

    for r, cls in enumerate(("rain", "sunny")):
        fname = keep.get(cls)
        img_path = ds.frames()[2] / cls / fname  # test dir
        if not img_path.is_file():
            img_path = ds.frames()[0] / cls / fname  # fallback all dir
        bgr = cv2.imread(str(img_path))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{cls.capitalize()} — {fname}", fontsize=10)
        axes[r, 0].axis("off")

        mask = examples.get(cls, {}).get(fname)
        if mask is not None:
            axes[r, 1].imshow(mask, cmap="gray")
            for cnt in contours:
                pts = cnt.squeeze()
                if pts.ndim == 2 and len(pts) >= 3:
                    pts = np.vstack([pts, pts[0]])
                    axes[r, 1].plot(pts[:, 0], pts[:, 1], color=BLUE, lw=1.5)
        axes[r, 1].set_title(f"MOG2 foreground ({cls})", fontsize=10)
        axes[r, 1].axis("off")

    fig.suptitle(f"MOG2 motion masks — {ds.camera}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ds.pictures / "_mog_masks.png", dpi=110)
    plt.close(fig)


def _write_mog_features(ds: Dataset, Xtr: np.ndarray, ytr: np.ndarray) -> None:
    """Save 2x3 grid of boxplots for MOG features on train set (rain vs sunny)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), facecolor=FACE)
    axes = axes.flatten()

    rain_idx = (ytr == 0)
    sunny_idx = (ytr == 1)

    for i, fname in enumerate(MOG_FEATURES):
        ax = axes[i]
        val_rain = Xtr[rain_idx, i]
        val_sunny = Xtr[sunny_idx, i]
        bp = ax.boxplot(
            [val_rain, val_sunny],
            tick_labels=["Rain", "Sunny"],
            patch_artist=True,
            medianprops=dict(color="black", lw=1.5),
        )
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][1].set_facecolor(GRAY)
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.set_title(fname, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"MOG2 motion features (train) — {ds.camera}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ds.pictures / "_mog_features.png", dpi=110)
    plt.close(fig)


def train_mog(ds: Dataset) -> dict:
    """Train MOG2 motion classifier + LogisticRegression for dataset."""
    _write_training_env(ds)
    ds.models.mkdir(parents=True, exist_ok=True)
    ds.pictures.mkdir(parents=True, exist_ok=True)

    all_dir, train_dir, test_dir = ds.frames()
    train_files = {
        cls: sorted((train_dir / cls).glob("*.jpg")) for cls in ("rain", "sunny")
    }
    test_files = {
        cls: sorted((test_dir / cls).glob("*.jpg")) for cls in ("rain", "sunny")
    }

    if not train_files["rain"] or not test_files["rain"]:
        _abort("no frames (run slice + split first)")

    keep = {
        "rain": test_files["rain"][len(test_files["rain"]) // 4].name,
        "sunny": test_files["sunny"][len(test_files["sunny"]) // 4].name,
    }

    print(f"Extracting MOG2 features for {ds.camera} (window={MOG_WINDOW})...")
    feat_dict, example_masks, extract_seconds = extract_mog_features(ds, keep=keep)
    print(f"Extracted MOG2 features in {extract_seconds:.2f}s")

    Xtr_list, ytr_list = [], []
    for cls, label in (("rain", 0), ("sunny", 1)):
        for p in train_files[cls]:
            Xtr_list.append(feat_dict[cls][p.name])
            ytr_list.append(label)

    Xte_list, yte_list, test_names = [], [], []
    for cls, label in (("rain", 0), ("sunny", 1)):
        for p in test_files[cls]:
            Xte_list.append(feat_dict[cls][p.name])
            yte_list.append(label)
            test_names.append(f"{cls}/{p.name}")

    Xtr = np.array(Xtr_list, dtype=np.float32)
    ytr = np.array(ytr_list, dtype=np.int64)
    Xte = np.array(Xte_list, dtype=np.float32)
    yte = np.array(yte_list, dtype=np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    t_tr0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_seconds = time.perf_counter() - t_tr0

    t_inf0 = time.perf_counter()
    proba = pipe.predict_proba(Xte)
    infer_seconds = time.perf_counter() - t_inf0

    p_rain = proba[:, 0]
    y_pred = proba.argmax(axis=1)

    joblib.dump(pipe, ds.models / "mog_logreg.joblib")

    coef = pipe["clf"].coef_[0]
    coef_sunny = {f: round(float(c), 4) for f, c in zip(MOG_FEATURES, coef)}

    params = {
        "subtractor": "cv2.createBackgroundSubtractorMOG2",
        "history": MOG_HISTORY,
        "varThreshold": MOG_VAR_THRESHOLD,
        "detectShadows": False,
        "window_frames": MOG_WINDOW,
        "work_size": f"{WORK_SIZE[0]}x{WORK_SIZE[1]}",
        "small_blob_area": [SMALL_BLOB_MIN, SMALL_BLOB_MAX],
        "classifier": "StandardScaler + LogisticRegression(C=1.0)",
        "coef_sunny": coef_sunny,
        "feature_extraction_seconds": round(extract_seconds, 2),
    }

    record = _build_metrics_record(
        "mog",
        yte,
        y_pred,
        p_rain,
        ds=ds,
        n_train=len(ytr),
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
        params=params,
        feature_importance=None,
        history=None,
        test_names=test_names,
    )

    _write_metrics_bars(
        ds.pictures / "_metrics_mog.png", record, f"MOG — test metrics ({ds.camera})"
    )
    _write_misclass_cm(
        ds.pictures / "_cm_mog.png",
        record["confusion"],
        f"MOG — confusion matrix ({ds.camera})",
    )
    coef_abs_sum = max(1e-6, float(np.sum(np.abs(coef))))
    feat_imp = {f: float(abs(c)) / coef_abs_sum for f, c in zip(MOG_FEATURES, coef)}
    _write_featimp(
        ds.pictures / "_featimp_mog.png",
        feat_imp,
        f"MOG — |logistic coefficient| (standardized) ({ds.camera})",
    )
    _write_mog_examples(ds, example_masks, keep)
    _write_mog_features(ds, Xtr, ytr)

    print(
        f"MOG test accuracy={record['accuracy']:.3f} "
        f"macro-F1={record['macro']['f1']:.3f} "
        f"AUC={record['roc_auc']:.3f}"
    )
    return record


if __name__ == "__main__":
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "kordon"
    train_mog(DATASETS[ds_key])
