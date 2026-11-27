"""Tabel supervised untuk model tabular (baseline, LightGBM, XGBoost).

Satu baris = (arah, titik asal i, horizon h). Titik asal hanya bin yang teramati (kamera sedang
merekam dan datanya layak); target y[i+h] juga harus teramati. Semua fitur hanya memakai data pada
waktu <= asal, kecuali kalender dan cuaca pada waktu target (skenario "ramalan cuaca sempurna").
Profil musiman dihitung ulang dari data sebelum `cutoff` agar tidak bocor.
"""
import numpy as np
import pandas as pd

from .common import ARAH, HORIZONS, STEPS_PER_DAY
from .siapkan_data import profil

FITUR_DASAR = [
    "arah_kode", "h_menit", "y0", "smp0", "lag1", "lag2", "lag3", "lag6", "lag12", "rm3", "rm6", "rm12",
    "n_obs12", "panjang_rekam", "lawan0", "lawan_rm6", "porsi_motor_rm12", "okupansi_rm6",
    "berhenti_rm6", "kemarin", "minggu_lalu", "profil_target", "profil_asal", "anomali0",
    "jam_target", "jam_sin", "jam_cos", "dow_target", "akhir_pekan", "jam_asal",
]
FITUR_CUACA_ASAL = ["hujan_asal", "hujan_1j_asal", "hujan_3j_asal", "suhu_asal", "rh_asal"]
FITUR_CUACA_TARGET = ["hujan_target", "hujan_1j_target", "suhu_target", "rh_target", "awan_target", "angin_target"]
SKENARIO = {
    "tanpa_cuaca": FITUR_DASAR,
    "cuaca_asal": FITUR_DASAR + FITUR_CUACA_ASAL,
    "cuaca_sempurna": FITUR_DASAR + FITUR_CUACA_ASAL + FITUR_CUACA_TARGET,
}


def _roll(x: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k] if k else x
    return out


def tabel(deret: pd.DataFrame, cuaca: pd.DataFrame, cutoff: pd.Timestamp, target: str = "flow",
          horizons: list[int] = HORIZONS, hanya_teramati: bool = True) -> pd.DataFrame:
    """hanya_teramati=False menyimpan juga baris yang targetnya tidak teramati (untuk meramal)."""
    deret = deret.copy()
    deret["prof"] = np.nan
    for arah in ARAH:      # profil musiman hanya dari data sebelum cutoff
        m = deret["arah"] == arah
        deret.loc[m, "prof"] = profil(deret[m].reset_index(drop=True), target, cutoff).to_numpy()
    cu = cuaca.set_index("waktu")
    rows = []
    for kode, arah in enumerate(ARAH):
        a = deret[deret["arah"] == arah].reset_index(drop=True)
        b = deret[deret["arah"] != arah].reset_index(drop=True)
        t = a["waktu"]
        x = a[f"{target}_isi"].to_numpy(float)
        y = a[target].to_numpy(float)
        obs = a["teramati"].to_numpy(bool)
        ada = ~np.isnan(x)
        run = pd.Series(ada).groupby((~pd.Series(ada)).cumsum()).cumsum().to_numpy()
        w = cu.reindex(t)
        motor_share = (a["motor_isi"] / a["aktif_isi"].where(a["aktif_isi"] > 0)).to_numpy(float)
        base = pd.DataFrame({
            "waktu_asal": t, "arah": arah, "arah_kode": kode,
            "y0": x, "smp0": a["smp_isi"].to_numpy(float) if target == "flow" else a["flow_isi"].to_numpy(float),
            **{f"lag{k}": _shift(x, k) for k in (1, 2, 3, 6, 12)},
            **{f"rm{k}": _roll(x, k) for k in (3, 6, 12)},
            "n_obs12": pd.Series(obs.astype(float)).rolling(12, min_periods=1).sum().to_numpy(),
            "panjang_rekam": run,
            "lawan0": b[f"{target}_isi"].to_numpy(float), "lawan_rm6": _roll(b[f"{target}_isi"].to_numpy(float), 6),
            "porsi_motor_rm12": _roll(motor_share, 12), "okupansi_rm6": _roll(a["okupansi_isi"].to_numpy(float), 6),
            "berhenti_rm6": _roll(a["rasio_berhenti_isi"].to_numpy(float), 6),
            "profil_asal": a["prof"].to_numpy(float), "jam_asal": (t.dt.hour + t.dt.minute / 60).to_numpy(),
            "hujan_asal": w["hujan"].to_numpy(), "hujan_1j_asal": w["hujan_1j"].to_numpy(),
            "hujan_3j_asal": w["hujan_3j"].to_numpy(), "suhu_asal": w["suhu"].to_numpy(), "rh_asal": w["rh"].to_numpy(),
        })
        base["anomali0"] = base["y0"] - base["profil_asal"]
        asal = np.flatnonzero(obs)
        for h in horizons:
            i = asal[asal + h < len(a)]
            j = i + h
            r = base.iloc[i].reset_index(drop=True)
            tj = t.iloc[j].reset_index(drop=True)
            wj = cu.reindex(tj)
            r["waktu_target"] = tj
            r["h"] = h
            r["h_menit"] = h * 5
            r["y"] = y[j]
            r["teramati_target"] = obs[j]
            for kol, mundur in (("kemarin", STEPS_PER_DAY), ("minggu_lalu", 7 * STEPS_PER_DAY)):
                k = j - mundur
                r[kol] = np.where(k >= 0, x[np.maximum(k, 0)], np.nan)
            r["profil_target"] = a["prof"].to_numpy(float)[j]
            jam = (tj.dt.hour + tj.dt.minute / 60).to_numpy()
            r["jam_target"] = jam
            r["jam_sin"], r["jam_cos"] = np.sin(2 * np.pi * jam / 24), np.cos(2 * np.pi * jam / 24)
            r["dow_target"] = tj.dt.dayofweek.to_numpy()
            r["akhir_pekan"] = (r["dow_target"] >= 5).astype(int)
            for k, src in zip(FITUR_CUACA_TARGET, ["hujan", "hujan_1j", "suhu", "rh", "awan", "angin"]):
                r[k] = wj[src].to_numpy()
            rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    if hanya_teramati:
        out = out[out["teramati_target"]]
    return out.drop(columns="teramati_target").reset_index(drop=True)


def baseline(tab: pd.DataFrame) -> pd.DataFrame:
    """Prediksi baseline pada baris tabel: persistence, seasonal naive (kemarin; bila kosong, profil), profil."""
    return pd.DataFrame({
        "persistence": tab["y0"],
        "seasonal_naive": tab["kemarin"].fillna(tab["profil_target"]),
        "profil_historis": tab["profil_target"],
    })
