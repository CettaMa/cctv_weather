"""Model deep learning dengan neuralforecast (GPU): LSTM, N-HiTS, TFT, TiDE, PatchTST, iTransformer.

Jalankan dari folder `extractions` (setelah model_gbdt):
    ..\\.venv\\Scripts\\python.exe -m forecast.model_deep [--max-steps 1500]

Model sekuens butuh input tanpa lubang. Celah panjang diisi profil musiman (dari data sebelum
cutoff) hanya sebagai input, dan kolom `available_mask` = bin teramati sehingga loss pelatihan tidak
pernah dihitung pada nilai isian. Kovariat historis: arus arah lawan dan jenis isian. Kovariat masa
depan: kalender dan cuaca (skenario ramalan cuaca sempurna). PatchTST dan iTransformer di
neuralforecast tidak menerima kovariat, jadi keduanya univariat/multivariat murni.

Dua varian panjang input (12 dan 24 jam) per model; varian dipilih dari MAE tahap validasi.
Ramalan dibuat untuk setiap titik asal (step 1 = 5 menit), lalu hanya baris yang sama dengan tabel
model tabular (asal dan target teramati) yang dinilai.
"""
import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
import torch

from .common import ARAH, H_MAX, HASIL, TRAIN_END, VAL_END, load_cuaca_5menit, load_deret
from .siapkan_data import profil

FUTR = ["jam_sin", "jam_cos", "dow_sin", "dow_cos", "akhir_pekan", "hujan", "hujan_1j", "suhu", "rh", "awan"]
HIST = ["lawan", "isian"]
INPUTS = [144, 288]
SKALA = 1000.0     # arus dibagi 1000 (kendaraan/jam -> ribu kendaraan/jam)


def frame(deret: pd.DataFrame, cuaca: pd.DataFrame, cutoff: pd.Timestamp, akhir: pd.Timestamp | None) -> pd.DataFrame:
    parts = {}
    for arah in ARAH:
        a = deret[deret["arah"] == arah].reset_index(drop=True)
        prof = profil(a, "flow", cutoff).to_numpy()
        y = a["flow_isi"].to_numpy(float)
        isian = np.where(a["teramati"], 0, np.where(np.isnan(y), 2, 1))
        parts[arah] = pd.DataFrame({"ds": a["waktu"], "y": np.where(np.isnan(y), prof, y),
                                    "available_mask": a["teramati"].astype(float), "isian": isian, "prof": prof})
    out = []
    for arah in ARAH:
        lawan = [x for x in ARAH if x != arah][0]
        p = parts[arah].copy()
        p["lawan"] = parts[lawan]["y"].to_numpy()
        p.insert(0, "unique_id", arah)
        if akhir is not None:
            p = p[p["ds"] < akhir]
        # tambah H_MAX langkah di ujung agar asal dekat akhir tetap mendapat ramalan horizon pendek
        pad_t = pd.date_range(p["ds"].iloc[-1] + pd.Timedelta("5min"), periods=H_MAX, freq="5min")
        pad = pd.DataFrame({"unique_id": arah, "ds": pad_t, "available_mask": 0.0, "isian": 2})
        tmp = pd.DataFrame({"waktu": pad_t, "arah": arah, "flow": np.nan})
        pad["y"] = profil(pd.concat([deret[deret["arah"] == arah][["waktu", "arah", "flow"]], tmp])
                          .reset_index(drop=True), "flow", cutoff).to_numpy()[-H_MAX:]
        pad["lawan"] = p["lawan"].iloc[-1]
        out.append(pd.concat([p.drop(columns="prof"), pad], ignore_index=True))
    df = pd.concat(out, ignore_index=True)
    t = df["ds"]
    jam = t.dt.hour + t.dt.minute / 60
    df["jam_sin"], df["jam_cos"] = np.sin(2 * np.pi * jam / 24), np.cos(2 * np.pi * jam / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * t.dt.dayofweek / 7), np.cos(2 * np.pi * t.dt.dayofweek / 7)
    df["akhir_pekan"] = (t.dt.dayofweek >= 5).astype(float)
    cu = cuaca.set_index("waktu").reindex(t)
    for k in ["hujan", "hujan_1j", "suhu", "rh", "awan"]:
        df[k] = cu[k].ffill().to_numpy()
    # Skala global tetap, bukan scaler per-jendela neuralforecast: scaler itu dihitung hanya dari titik
    # teramati, dan jendela yang hampir seluruhnya isian membuat skalanya ~0 sehingga ramalan meledak.
    df["y"] = df["y"] / SKALA
    df["lawan"] = df["lawan"] / SKALA
    df["suhu"] = (df["suhu"] - 25) / 5
    df["rh"] = df["rh"] / 100
    df["awan"] = df["awan"] / 100
    df["ds"] = df["ds"].dt.tz_localize(None)
    return df


def models(input_size: int, max_steps: int) -> list:
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM, NHITS, TFT, PatchTST, TiDE, iTransformer
    common = dict(h=H_MAX, input_size=input_size, max_steps=max_steps, loss=MAE(), valid_loss=MAE(),
                  scaler_type="identity", early_stop_patience_steps=10, val_check_steps=50, random_seed=1,
                  enable_progress_bar=False, enable_model_summary=False, logger=False)
    ex = dict(futr_exog_list=FUTR, hist_exog_list=HIST)
    # batch jendela kecil: VRAM 6 GB (TFT dengan 1024 jendela x 24 jam kehabisan memori)
    uni = dict(windows_batch_size=256, inference_windows_batch_size=128)
    s = f"_in{input_size}"
    return [
        LSTM(**common, **ex, **uni, encoder_hidden_size=128, decoder_hidden_size=128, learning_rate=1e-3,
             alias="lstm" + s),
        NHITS(**common, **ex, **uni, learning_rate=1e-3, alias="nhits" + s),
        TFT(**common, **ex, windows_batch_size=64, inference_windows_batch_size=32, hidden_size=32,
            learning_rate=1e-3, alias="tft" + s),
        TiDE(**common, **ex, **uni, learning_rate=1e-3, alias="tide" + s),
        PatchTST(**common, **uni, patch_len=24, stride=12, learning_rate=1e-3, alias="patchtst" + s),
        iTransformer(**common, n_series=2, learning_rate=1e-3, alias="itransformer" + s),
    ]


def run(df: pd.DataFrame, mulai: pd.Timestamp, max_steps: int, nama: str) -> pd.DataFrame:
    """Latih dan ramal satu model per kali; hasil tiap model disimpan agar bisa dilanjutkan."""
    import time
    from neuralforecast import NeuralForecast
    cache = HASIL / "deep_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ds0 = mulai.tz_localize(None)
    L = df["unique_id"].value_counts().iloc[0]
    idx = int((df[df["unique_id"] == ARAH[0]]["ds"] < ds0).sum())
    test_size = L - idx
    res = []
    for inp in INPUTS:
        for model in models(inp, max_steps):
            f = cache / f"{nama}_{model.alias}.parquet"
            if not f.exists():
                t0 = time.time()
                nf = NeuralForecast(models=[model], freq="5min")
                cv = nf.cross_validation(df=df, step_size=1, val_size=2 * 288, test_size=test_size, refit=False,
                                         n_windows=None)
                cv[model.alias] = (cv[model.alias] * SKALA).clip(lower=0)     # arus tidak negatif
                cv.drop(columns="y").to_parquet(f, index=False)
                print(f"{nama} {model.alias}: {time.time() - t0:.0f} s", flush=True)
                del nf
                torch.cuda.empty_cache()
            res.append(pd.read_parquet(f).set_index(["unique_id", "ds", "cutoff"]))
    return pd.concat(res, axis=1).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    torch.set_float32_matmul_precision("high")
    deret, cuaca = load_deret(), load_cuaca_5menit()
    for nama, cutoff, akhir in [("validasi", TRAIN_END, VAL_END), ("uji", VAL_END, None)]:
        df = frame(deret, cuaca, cutoff, akhir)
        cv = run(df, cutoff, args.max_steps, nama)
        cv = cv.rename(columns={"unique_id": "arah", "cutoff": "waktu_asal", "ds": "waktu_target"})
        cv.to_parquet(HASIL / f"pred_deep_{nama}.parquet", index=False)
        print(nama, cv.shape, flush=True)


if __name__ == "__main__":
    main()
