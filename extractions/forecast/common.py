"""Konstanta dan utilitas bersama untuk forecasting arus lalu lintas."""
from pathlib import Path

import numpy as np
import pandas as pd

F = Path(__file__).resolve().parent
ROOT = F.parent
DATA = F / "data"
HASIL = F / "hasil"
GAMBAR = F / "gambar"
SUMBER = ROOT / "eda" / "data" / "dataset_menit_berflag.csv"
DERET = DATA / "deret_5menit.csv"
ULANG = ROOT / "eda" / "data" / "ulang_per_menit.csv"   # dibuat eda/pindai_ulang.py

TZ = "Asia/Jakarta"
STEP = pd.Timedelta("5min")
STEPS_PER_DAY = 288
# horizon dalam langkah 5 menit: 5 menit, lalu 15, 30, ..., 360 menit
HORIZONS = [1] + list(range(3, 73, 3))
H_MAX = max(HORIZONS)
MAX_FILL = 6                      # celah <= 6 langkah (30 menit) boleh diisi
ARAH = ["menjauhi", "mendekati"]
TARGET = "flow"                   # est_flow_veh_per_hour
TARGET2 = "smp"                   # est_q_smp_per_hour

# split kronologis (waktu lokal); target pada waktu >= batas masuk bagian berikutnya.
# Dipilih agar bin teramati kira-kira 73/11/16%: 21-22 dan 24-26 Nov banyak menit tidak layak.
TRAIN_END = pd.Timestamp("2026-11-20 00:00", tz=TZ)
VAL_END = pd.Timestamp("2026-11-23 00:00", tz=TZ)

LAT, LON = -6.9652846, 107.6381471   # titik CCTV Terusan Buahbatu

CUACA_KOLOM = ["hujan", "suhu", "rh", "awan", "angin"]


def bagian(t: pd.Series) -> pd.Series:
    """latih / validasi / uji menurut waktu."""
    return pd.Series(np.select([t < TRAIN_END, t < VAL_END], ["latih", "validasi"], "uji"), index=t.index)


def load_deret() -> pd.DataFrame:
    d = pd.read_csv(DERET)
    d["waktu"] = pd.to_datetime(d["waktu"], utc=True).dt.tz_convert(TZ)
    return d


def load_cuaca_5menit() -> pd.DataFrame:
    c = pd.read_csv(DATA / "cuaca_5menit.csv")
    c["waktu"] = pd.to_datetime(c["waktu"], utc=True).dt.tz_convert(TZ)
    return c


def metrik(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "WAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100)}
