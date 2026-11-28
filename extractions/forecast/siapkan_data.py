"""Ubah dataset per menit (hasil EDA) menjadi deret 5 menit reguler per arah, dengan imputasi terbatas.

Jalankan dari folder `extractions`:
    ..\\.venv\\Scripts\\python.exe -m forecast.siapkan_data

Aturan:
- Menit yang berisi rekaman ulangan (eda/pindai_ulang.py) tidak dianggap layak.
- Hanya menit `layak` yang dipakai. Satu bin 5 menit dianggap teramati bila punya >= 3 menit layak;
  nilainya rata-rata berbobot detik sampel.
- Celah <= 30 menit (6 bin) di antara dua bin teramati diisi: profil musiman + interpolasi linear
  residual. Celah lebih panjang dibiarkan kosong. Kolom `y` (target) tidak pernah diisi; nilai isian
  hanya ada di kolom `*_isi` untuk input model.
- Profil musiman (jam x hari kerja/akhir pekan) dihitung dari bagian latih saja.
"""
import json

import numpy as np
import pandas as pd

from .common import ARAH, DATA, DERET, MAX_FILL, SUMBER, TRAIN_END, TZ, ULANG, bagian

KOLOM = {
    "est_flow_veh_per_hour": "flow", "est_q_smp_per_hour": "smp",
    "average_active_motor": "motor", "average_active_mobil": "mobil", "average_active_bus": "bus",
    "average_active_truk": "truk", "average_active_total": "aktif", "avg_roi_occupancy_pct": "okupansi",
    "avg_stopped_ratio": "rasio_berhenti", "avg_dj_effective": "dj",
}
ISI = ["flow", "smp", "motor", "mobil", "aktif", "okupansi", "rasio_berhenti"]
MIN_MENIT = 3


def baca() -> pd.DataFrame:
    d = pd.read_csv(SUMBER)
    d["menit"] = pd.to_datetime(d["menit"], utc=True).dt.tz_convert(TZ)
    for c in [c for c in d.columns if c.startswith("flag_")] + ["layak"]:
        d[c] = d[c].astype(bool)
    d["arah"] = d["arah"].astype("category")
    d["one_min_traffic_condition"] = d["one_min_traffic_condition"].astype("category")
    num = d.select_dtypes("number").columns
    d[num] = d[num].astype("float32")
    d["ulang"] = d["menit"].isin(menit_ulang())
    d["layak"] &= ~d["ulang"]
    return d


def menit_ulang() -> pd.Series:
    """Menit yang lebih dari separuh sampel jam overlay-nya sudah pernah terekam."""
    u = pd.read_csv(ULANG)
    u = u[(u["porsi_ulang"] > 0.5) & (u["terbaca"] >= 2)]
    return pd.to_datetime(u["menit"], utc=True).dt.tz_convert(TZ)


def agregasi(d: pd.DataFrame) -> pd.DataFrame:
    grid = pd.date_range(d["menit"].min().floor("5min"), d["menit"].max().floor("5min"), freq="5min")
    out = []
    for arah in ARAH:
        a = d[(d["arah"] == arah) & d["layak"]].copy()
        a["bin"] = a["menit"].dt.floor("5min")
        w = a["actual_sample_duration_seconds"]
        g = pd.DataFrame({"n_menit": a.groupby("bin").size()})
        for src, dst in KOLOM.items():
            g[dst] = (a[src] * w).groupby(a["bin"]).sum() / w.groupby(a["bin"]).sum()
        semua = d[d["arah"] == arah].groupby(d["menit"].dt.floor("5min")).size()
        g = g.reindex(grid)
        g["n_menit_rekam"] = semua.reindex(grid).fillna(0).astype(int).to_numpy()
        g["n_menit"] = g["n_menit"].fillna(0).astype(int)
        kurang = g["n_menit"] < MIN_MENIT
        g.loc[kurang, list(KOLOM.values())] = np.nan
        g["teramati"] = ~kurang
        n_ulang = d[(d["arah"] == arah) & d["ulang"]].groupby(d["menit"].dt.floor("5min")).size()
        g["n_menit_ulang"] = n_ulang.reindex(grid).fillna(0).astype(int).to_numpy()
        g.insert(0, "arah", arah)
        out.append(g.rename_axis("waktu").reset_index())
    return pd.concat(out, ignore_index=True)


def profil(df: pd.DataFrame, kolom: str, cutoff: pd.Timestamp = TRAIN_END) -> pd.Series:
    """Median per (akhir pekan, jam) dari data sebelum `cutoff`, dipetakan ke tiap baris; dihaluskan antar-jam."""
    t = df["waktu"]
    key = pd.DataFrame({"we": (t.dt.dayofweek >= 5).astype(int), "jam": t.dt.hour, "arah": df["arah"]})
    latih = (t < cutoff) & df[kolom].notna()
    tab = df.loc[latih, kolom].groupby([key.loc[latih, "arah"], key.loc[latih, "we"], key.loc[latih, "jam"]]).median()
    full = pd.MultiIndex.from_product([ARAH, [0, 1], range(24)], names=["arah", "we", "jam"])
    tab = tab.reindex(full)
    # jam yang tidak ada di latih: ambil dari jenis hari lain, lalu haluskan melingkar 3 jam
    tab = tab.fillna(tab.groupby(level=["arah", "jam"]).transform("mean"))
    tab = tab.groupby(level=["arah", "we"], group_keys=False).apply(
        lambda s: pd.Series(np.convolve(np.r_[s.values[-1], s.values, s.values[0]], [0.25, 0.5, 0.25], "valid"),
                            index=s.index))
    # interpolasi linear antar-jam pada menit (jam + menit/60)
    jam_f = t.dt.hour + t.dt.minute / 60
    lo = np.floor(jam_f - 0.5).astype(int) % 24
    hi = (lo + 1) % 24
    frac = (jam_f - 0.5) % 1
    v_lo = tab.reindex(pd.MultiIndex.from_arrays([df["arah"], key["we"], lo])).to_numpy()
    v_hi = tab.reindex(pd.MultiIndex.from_arrays([df["arah"], key["we"], hi])).to_numpy()
    return pd.Series(v_lo * (1 - frac.to_numpy()) + v_hi * frac.to_numpy(), index=df.index)


def isi_celah(s: pd.Series, prof: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Isi celah <= MAX_FILL yang diapit nilai teramati; kembalikan (terisi, panjang_celah, langkah_sejak_obs)."""
    obs = s.notna().to_numpy()
    n = len(s)
    run_id = np.cumsum(np.r_[True, obs[1:] != obs[:-1]])
    panjang = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    panjang = np.where(obs, 0, panjang)
    idx = np.arange(n)
    prev_obs = np.maximum.accumulate(np.where(obs, idx, -1))
    nxt = np.where(obs, idx, n)[::-1]
    next_obs = np.minimum.accumulate(nxt)[::-1]
    diapit = (~obs) & (prev_obs >= 0) & (next_obs < n)
    boleh = diapit & (panjang <= MAX_FILL)
    resid = (s - prof).interpolate(limit_area="inside")
    hasil = s.astype("float64")
    hasil[boleh] = (prof + resid)[boleh]
    sejak = np.where(prev_obs >= 0, idx - prev_obs, -1)
    return hasil, pd.Series(panjang, index=s.index), pd.Series(sejak, index=s.index)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    d = baca()
    g = agregasi(d)
    ringkas = {"bin_total_per_arah": int(len(g) // 2)}
    parts = []
    for arah in ARAH:
        a = g[g["arah"] == arah].reset_index(drop=True)
        for k in ISI:
            prof = profil(a, k)
            a[f"{k}_isi"], pc, sejak = isi_celah(a[k], prof)
            a[f"{k}_profil"] = prof
            if k == "flow":
                a["panjang_celah"], a["langkah_sejak_obs"] = pc, sejak
        a["diimputasi"] = a["flow"].isna() & a["flow_isi"].notna()
        a["bagian"] = bagian(a["waktu"])
        ringkas[arah] = {
            "teramati": int(a["teramati"].sum()),
            "diimputasi": int(a["diimputasi"].sum()),
            "kosong": int(a["flow_isi"].isna().sum()),
            "teramati_per_bagian": a[a["teramati"]].groupby("bagian").size().to_dict(),
        }
        parts.append(a)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(DERET, index=False)
    (DATA / "ringkasan_deret.json").write_text(json.dumps(ringkas, indent=1), encoding="utf-8")
    print(json.dumps(ringkas, indent=1))


if __name__ == "__main__":
    main()
