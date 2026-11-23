#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expanded tabular feature extractor (24 features) and training pipeline for
7 tabular models:
    Machine learning: Random Forest, XGBoost, LightGBM, CatBoost
    Deep learning:    MLP, TabNet, FT-Transformer

Evaluated on the SP Buah Batu weather dataset with binary label:
    0 = rain (positive)
    1 = sunny
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from script_tools.weather_vision import (
    extract_frame_features,
    roi_mask,
    WORK_SIZE,
    SPEC_REF,
)
from script_tools.weather_ml import (
    Dataset,
    DATASETS,
    _abort,
    _write_training_env,
    _build_metrics_record,
    _write_metrics_bars,
    _write_misclass_cm,
    _write_featimp,
    _write_curves,
)

TAB_FEATURES = [
    # 4 original calibration features
    "spec_ratio",
    "road_gray",
    "sat_mean",
    "lap_var",
    # HSV channel statistics inside ROI
    "hue_mean",
    "hue_std",
    "sat_std",
    "val_mean",
    "val_std",
    # RGB channel statistics inside ROI
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    # Gray-level percentiles inside ROI
    "gray_p10",
    "gray_p50",
    "gray_p90",
    # Gray contrast inside ROI
    "gray_std",
    # Texture & edge density inside ROI
    "edge_density",
    "dark_ratio",
    "spec_ratio_roi",
    "sobel_mag_mean",
    "blue_ratio",
]

TAB_FEATURE_DESC = {
    "spec_ratio": "Full-frame specular highlight ratio (fraction of gray pixels > 200)",
    "road_gray": "Mean grayscale intensity inside road ROI",
    "sat_mean": "Mean HSV saturation inside road ROI",
    "lap_var": "Variance of Laplacian inside road ROI (focus/texture cue)",
    "hue_mean": "Mean HSV hue inside road ROI",
    "hue_std": "Standard deviation of HSV hue inside road ROI",
    "sat_std": "Standard deviation of HSV saturation inside road ROI",
    "val_mean": "Mean HSV brightness/value inside road ROI",
    "val_std": "Standard deviation of HSV brightness inside road ROI",
    "r_mean": "Mean red channel intensity inside road ROI",
    "g_mean": "Mean green channel intensity inside road ROI",
    "b_mean": "Mean blue channel intensity inside road ROI",
    "r_std": "Standard deviation of red channel inside road ROI",
    "g_std": "Standard deviation of green channel inside road ROI",
    "b_std": "Standard deviation of blue channel inside road ROI",
    "gray_p10": "10th percentile of grayscale intensity inside road ROI",
    "gray_p50": "Median (50th percentile) grayscale intensity inside road ROI",
    "gray_p90": "90th percentile of grayscale intensity inside road ROI",
    "gray_std": "Standard deviation of grayscale intensity inside road ROI (contrast)",
    "edge_density": "Fraction of Canny edge pixels inside road ROI",
    "dark_ratio": "Fraction of dark pixels (gray < 50) inside road ROI",
    "spec_ratio_roi": "Fraction of wet specular highlight pixels (gray > 200) inside road ROI",
    "sobel_mag_mean": "Mean Sobel gradient magnitude inside road ROI",
    "blue_ratio": "Blue chromaticity fraction (B / (R + G + B)) inside road ROI",
}


def extract_tabular_features(frame: np.ndarray, ds: Dataset | None = None) -> dict[str, float]:
    """Extract 24 tabular cues from a BGR video frame."""
    small = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    if ds is not None:
        mask = roi_mask(sh, sw, roi_rel=ds.feat_roi, text_top=ds.feat_text_top, text_bottom=ds.feat_text_bottom)
        base = extract_frame_features(frame, roi_rel=ds.feat_roi, text_top=ds.feat_text_top, text_bottom=ds.feat_text_bottom)
    else:
        mask = roi_mask(sh, sw)
        base = extract_frame_features(frame)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    roi_gray = gray[mask]
    roi_hsv = hsv[mask]
    roi_rgb = rgb[mask]

    feats: dict[str, float] = {
        "spec_ratio": float(base["spec_ratio"]),
        "road_gray": float(base["road_gray"]),
        "sat_mean": float(base["sat_mean"]),
        "lap_var": float(base["lap_var"]),
    }
    if roi_gray.size == 0:
        for k in TAB_FEATURES[4:]:
            feats[k] = 0.0
        return feats

    # HSV stats
    h_channel = roi_hsv[:, 0].astype(np.float32)
    s_channel = roi_hsv[:, 1].astype(np.float32)
    v_channel = roi_hsv[:, 2].astype(np.float32)

    feats["hue_mean"] = float(np.mean(h_channel))
    feats["hue_std"] = float(np.std(h_channel))
    feats["sat_std"] = float(np.std(s_channel))
    feats["val_mean"] = float(np.mean(v_channel))
    feats["val_std"] = float(np.std(v_channel))

    # RGB stats
    r_chan = roi_rgb[:, 0].astype(np.float32)
    g_chan = roi_rgb[:, 1].astype(np.float32)
    b_chan = roi_rgb[:, 2].astype(np.float32)

    feats["r_mean"] = float(np.mean(r_chan))
    feats["g_mean"] = float(np.mean(g_chan))
    feats["b_mean"] = float(np.mean(b_chan))
    feats["r_std"] = float(np.std(r_chan))
    feats["g_std"] = float(np.std(g_chan))
    feats["b_std"] = float(np.std(b_chan))

    # Gray percentiles & contrast
    rg_float = roi_gray.astype(np.float32)
    feats["gray_p10"] = float(np.percentile(rg_float, 10))
    feats["gray_p50"] = float(np.percentile(rg_float, 50))
    feats["gray_p90"] = float(np.percentile(rg_float, 90))
    feats["gray_std"] = float(np.std(rg_float))

    # Edge density
    canny = cv2.Canny(gray, 100, 200)
    feats["edge_density"] = float(np.mean(canny[mask] > 0))

    # Dark ratio & specular inside ROI
    feats["dark_ratio"] = float(np.mean(roi_gray < 50))
    feats["spec_ratio_roi"] = float(np.mean(roi_gray > SPEC_REF))

    # Sobel gradient magnitude
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    feats["sobel_mag_mean"] = float(np.mean(sobel_mag[mask]))

    # Blue chromaticity ratio
    total_rgb = feats["r_mean"] + feats["g_mean"] + feats["b_mean"] + 1e-6
    feats["blue_ratio"] = float(feats["b_mean"] / total_rgb)

    return feats


def load_tabular(split_dir: Path, ds: Dataset | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load tabular feature matrix X, binary labels y, and list of frame relative paths.
        y: 0 = rain, 1 = sunny
    """
    xs, ys, names = [], [], []
    for cls, label in (("rain", 0), ("sunny", 1)):
        cls_dir = split_dir / cls
        paths = sorted(cls_dir.glob("*.jpg"))
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            feats = extract_tabular_features(frame, ds=ds)
            xs.append([feats[k] for k in TAB_FEATURES])
            ys.append(label)
            names.append(f"{cls}/{p.name}")

    if not xs:
        _abort(f"no frames in {split_dir} (run slice + split first)")

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64), names

# --------------------------------------------------------------------------- #
# Deep learning architectures (PyTorch)
# --------------------------------------------------------------------------- #

class _FTTransformer:
    pass  # Defined dynamically if torch is available


def _get_ft_transformer_cls():
    import torch
    import torch.nn as nn

    class FTTransformer(nn.Module):
        """
        Feature Tokenizer Transformer for tabular data (Gorishniy et al., 2021).
        """
        def __init__(
            self,
            n_features: int = 24,
            d_token: int = 64,
            n_blocks: int = 3,
            n_heads: int = 8,
            ffn_dim: int = 128,
            dropout: float = 0.1,
            num_classes: int = 2,
        ):
            super().__init__()
            self.n_features = n_features
            self.d_token = d_token

            # Feature tokenizer: weight and bias per feature
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))

            d_sqrt_inv = 1.0 / np.sqrt(d_token)
            nn.init.uniform_(self.weight, -d_sqrt_inv, d_sqrt_inv)
            nn.init.uniform_(self.bias, -d_sqrt_inv, d_sqrt_inv)
            nn.init.uniform_(self.cls_token, -d_sqrt_inv, d_sqrt_inv)

            layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=n_blocks,
                enable_nested_tensor=False,
            )

            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.ReLU(),
                nn.Linear(d_token, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch_size, n_features)
            b = x.size(0)
            # (batch, n_features, 1) * (1, n_features, d_token) + (1, n_features, d_token)
            tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features + 1, d_token)

            encoded = self.transformer(tokens)
            cls_out = encoded[:, 0, :]  # CLS token output
            return self.head(cls_out)

    return FTTransformer


def _fit_torch_tabular(
    model,
    Xf: np.ndarray,
    yf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    *,
    optimizer,
    epochs: int = 200,
    batch_size: int = 16,
    patience: int = 25,
) -> dict:
    """Shared PyTorch training loop with early stopping for tabular neural models."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(42)
    train_ds = TensorDataset(torch.from_numpy(Xf), torch.from_numpy(yf))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    xv_t = torch.from_numpy(Xv)
    yv_t = torch.from_numpy(yv)

    criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_ld:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(Xf)

        model.eval()
        with torch.no_grad():
            out_v = model(xv_t)
            val_loss = criterion(out_v, yv_t).item()
            pred_v = out_v.argmax(dim=1)
            val_acc = (pred_v == yv_t).float().mean().item()

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    return hist


# --------------------------------------------------------------------------- #
# Tabular models training dispatcher
# --------------------------------------------------------------------------- #

def train_tabular(key: str, ds: Dataset | None = None) -> dict:
    """Train one of the 7 tabular models on the expanded 24-feature dataset."""
    if ds is None:
        ds = DATASETS["sp_buahbatu"]

    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass

    _write_training_env(ds)
    ds.models.mkdir(parents=True, exist_ok=True)
    ds.pictures.mkdir(parents=True, exist_ok=True)

    _, train_dir, test_dir = ds.frames()
    Xtr, ytr, _ = load_tabular(train_dir, ds=ds)
    Xte, yte, test_names = load_tabular(test_dir, ds=ds)

    n_tr = len(Xtr)
    n_te = len(Xte)
    train_start = time.perf_counter()
    importances: dict[str, float] | None = None
    history: dict | None = None
    params: dict = {}

    if key == "rf":
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        rf = RandomForestClassifier(**params)
        rf.fit(Xtr, ytr)
        train_seconds = time.perf_counter() - train_start

        joblib.dump(rf, str(ds.models / "rf_weather.joblib"))
        infer_start = time.perf_counter()
        proba = rf.predict_proba(Xte)
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

        raw_imp = rf.feature_importances_
        s = float(np.sum(raw_imp))
        norm_imp = (raw_imp / s) if s > 0 else raw_imp
        importances = {feat: float(norm_imp[i]) for i, feat in enumerate(TAB_FEATURES)}

    elif key == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            _abort("xgboost not installed (.venv/Scripts/python.exe -m pip install xgboost)")
        import joblib

        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        xgb = XGBClassifier(**params)
        xgb.fit(Xtr, ytr)
        train_seconds = time.perf_counter() - train_start

        joblib.dump(xgb, str(ds.models / "xgb_weather.joblib"))
        infer_start = time.perf_counter()
        proba = xgb.predict_proba(Xte)
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

        raw_imp = xgb.feature_importances_
        s = float(np.sum(raw_imp))
        norm_imp = (raw_imp / s) if s > 0 else raw_imp
        importances = {feat: float(norm_imp[i]) for i, feat in enumerate(TAB_FEATURES)}

    elif key == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            _abort("lightgbm not installed (.venv/Scripts/python.exe -m pip install lightgbm)")
        import joblib

        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbose": -1,
        }
        lgbm = LGBMClassifier(**params)
        lgbm.fit(Xtr, ytr)
        train_seconds = time.perf_counter() - train_start

        joblib.dump(lgbm, str(ds.models / "lgbm_weather.joblib"))
        infer_start = time.perf_counter()
        proba = lgbm.predict_proba(Xte)
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

        raw_imp = lgbm.feature_importances_.astype(np.float64)
        s = float(np.sum(raw_imp))
        norm_imp = (raw_imp / s) if s > 0 else raw_imp
        importances = {feat: float(norm_imp[i]) for i, feat in enumerate(TAB_FEATURES)}

    elif key == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            _abort("catboost not installed (.venv/Scripts/python.exe -m pip install catboost)")
        import joblib

        params = {
            "iterations": 500,
            "depth": 4,
            "learning_rate": 0.05,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        cat = CatBoostClassifier(**params)
        cat.fit(Xtr, ytr)
        train_seconds = time.perf_counter() - train_start

        joblib.dump(cat, str(ds.models / "catboost_weather.joblib"))
        infer_start = time.perf_counter()
        proba = cat.predict_proba(Xte)
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

        raw_imp = np.array(cat.get_feature_importance(), dtype=np.float64)
        s = float(np.sum(raw_imp))
        norm_imp = (raw_imp / s) if s > 0 else raw_imp
        importances = {feat: float(norm_imp[i]) for i, feat in enumerate(TAB_FEATURES)}

    elif key in ("mlp", "ftt"):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib

        scaler = StandardScaler()
        Xtr_norm = scaler.fit_transform(Xtr)
        Xte_norm = scaler.transform(Xte)
        joblib.dump(scaler, str(ds.models / "tabular_scaler.joblib"))
        idx_fit, idx_val = train_test_split(
            np.arange(n_tr), test_size=0.15, random_state=42, stratify=ytr
        )
        Xf, yf = Xtr_norm[idx_fit], ytr[idx_fit]
        Xv, yv = Xtr_norm[idx_val], ytr[idx_val]

        if key == "mlp":
            model = nn.Sequential(
                nn.Linear(24, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2),
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            params = {
                "arch": "Linear(24,64)-ReLU-Drop0.2-Linear(64,32)-ReLU-Drop0.2-Linear(32,2)",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "patience": 25,
            }
        else:  # ftt
            FTTransformer = _get_ft_transformer_cls()
            model = FTTransformer(
                n_features=24,
                d_token=64,
                n_blocks=3,
                n_heads=8,
                ffn_dim=128,
                dropout=0.1,
                num_classes=2,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            params = {
                "arch": "FT-Transformer(d_token=64,n_blocks=3,n_heads=8,ffn_dim=128,drop=0.1)",
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "batch_size": 16,
                "patience": 25,
            }

        history = _fit_torch_tabular(
            model, Xf, yf, Xv, yv, optimizer=opt, epochs=200, batch_size=16, patience=25
        )
        train_seconds = time.perf_counter() - train_start
        params["epochs_trained"] = len(history["train_loss"])

        torch.save(model.state_dict(), str(ds.models / f"{key}_weather.pt"))
        model.eval()
        infer_start = time.perf_counter()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte_norm))
            proba = torch.softmax(logits, dim=-1).numpy()
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

    elif key == "tabnet":
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            _abort("pytorch_tabnet not installed (.venv/Scripts/python.exe -m pip install pytorch-tabnet)")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib

        scaler = StandardScaler()
        Xtr_norm = scaler.fit_transform(Xtr)
        Xte_norm = scaler.transform(Xte)
        joblib.dump(scaler, str(ds.models / "tabular_scaler.joblib"))

        idx_fit, idx_val = train_test_split(
            np.arange(n_tr), test_size=0.15, random_state=42, stratify=ytr
        )
        Xf, yf = Xtr_norm[idx_fit], ytr[idx_fit]
        Xv, yv = Xtr_norm[idx_val], ytr[idx_val]

        params = {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "seed": 42,
            "optimizer_params": {"lr": 2e-2},
            "verbose": 0,
        }
        tabnet = TabNetClassifier(**params)
        tabnet.fit(
            Xf,
            yf,
            eval_set=[(Xv, yv)],
            eval_name=["val"],
            eval_metric=["accuracy", "logloss"],
            max_epochs=200,
            patience=30,
            batch_size=64,
            virtual_batch_size=16,
            drop_last=False,
        )
        train_seconds = time.perf_counter() - train_start

        tabnet.save_model(str(ds.models / "tabnet_weather"))
        infer_start = time.perf_counter()
        proba = tabnet.predict_proba(Xte_norm)
        infer_seconds = time.perf_counter() - infer_start

        p_rain = proba[:, 0]
        y_pred = proba.argmax(axis=1)

        raw_imp = tabnet.feature_importances_
        s = float(np.sum(raw_imp))
        norm_imp = (raw_imp / s) if s > 0 else raw_imp
        importances = {feat: float(norm_imp[i]) for i, feat in enumerate(TAB_FEATURES)}

        history = {
            "train_loss": [float(x) for x in tabnet.history["loss"]],
            "val_loss": [float(x) for x in tabnet.history["val_logloss"]],
            "val_acc": [float(x) for x in tabnet.history["val_accuracy"]],
        }
        params["epochs_trained"] = len(history["train_loss"])

    else:
        _abort(f"unknown tabular model key: {key}")

    record = _build_metrics_record(
        key,
        yte,
        y_pred,
        p_rain,
        ds=ds,
        n_train=n_tr,
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
        params=params,
        feature_importance=importances,
        history=history,
        test_names=test_names,
    )

    display_name = record["display_name"]
    _write_metrics_bars(
        ds.pictures / f"_metrics_{key}.png",
        record,
        f"{display_name} — test metrics ({ds.camera})",
    )
    _write_misclass_cm(
        ds.pictures / f"_cm_{key}.png",
        record["confusion"],
        f"{display_name} — confusion matrix ({ds.camera})",
    )

    if importances is not None:
        _write_featimp(
            ds.pictures / f"_featimp_{key}.png",
            importances,
            f"{display_name} — feature importances ({ds.camera})",
        )

    if history is not None:
        _write_curves(
            history,
            ds.pictures / f"_curves_{key}.png",
            f"{display_name} training curves ({ds.camera})",
        )

    print(
        f"{display_name} test accuracy={record['accuracy']:.3f} "
        f"macro-F1={record['macro']['f1']:.3f} "
        f"AUC={record['roc_auc']:.3f}"
    )
    return record


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_ds = DATASETS[sys.argv[2]] if len(sys.argv) > 2 else DATASETS["sp_buahbatu"]
        train_tabular(sys.argv[1], target_ds)
    else:
        print("Usage: python -m script_tools.weather_tabular <model_key> [sp_buahbatu|kordon]")
