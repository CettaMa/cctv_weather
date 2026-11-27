"""Apakah arus berubah beberapa menit SETELAH cuaca berubah?

Jalankan dari folder `extractions`:
    ..\\.venv\\Scripts\\python.exe -m forecast.jeda_cuaca

1. Korelasi silang: residual arus 5 menit (terhadap profil jam x jenis hari) vs hujan yang digeser
   -60 sampai +180 menit. Jeda positif = hujan mendahului arus.
2. Kejadian awal hujan: jam hujan >= AMBANG setelah jam kering (< 0,1 mm), lalu residual arus dari
   60 menit sebelum sampai 120 menit sesudah awal jam hujan, bila kamera merekam.
Dilakukan untuk OpenWeatherMap dan Open-Meteo. Keduanya per JAM, jadi saat hujan mulai di dalam jam
itu tidak diketahui; jeda 5-20 menit tidak bisa dipisahkan dari data ini saja.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from .common import ARAH, DATA, HASIL, TZ, load_deret
from .cuaca import ke_5menit
from .gaya import ARAH_NAMA, ARAH_WARNA, INK2, MUTED, plt, simpan
from .siapkan_data import profil

LAGS = range(-12, 37)          # langkah 5 menit: -60 ... +180 menit
AMBANG = 0.5
SUMBER = {"owm": "OpenWeatherMap", "openmeteo": "Open-Meteo"}


def residual(deret: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arah in ARAH:
        a = deret[deret["arah"] == arah].reset_index(drop=True)
        a["res"] = a["flow"] - profil(a, "flow", pd.Timestamp.max.tz_localize("UTC"))
        out.append(a[["waktu", "arah", "res", "teramati"]])
    return pd.concat(out, ignore_index=True)


def hujan_5menit(sumber: str, waktu: pd.Series) -> pd.Series:
    raw = pd.read_csv(DATA / f"cuaca_{sumber}.csv")
    c = ke_5menit(raw, waktu.min(), waktu.max()).set_index("waktu")["hujan"]
    return c


def main() -> None:
    res = residual(load_deret())
    hasil = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    for col, (s, nama) in enumerate(SUMBER.items()):
        hujan = hujan_5menit(s, res["waktu"])
        hasil[s] = {"ccf": {}, "kejadian": []}
        ax = axes[0, col]
        for arah in ARAH:
            a = res[res["arah"] == arah].set_index("waktu")
            r = a["res"]
            rows = []
            for L in LAGS:
                h = hujan.shift(L).reindex(r.index)
                m = r.notna() & h.notna()
                rho = stats.spearmanr(r[m], h[m])[0]
                rows.append((L * 5, rho, int(m.sum())))
            ccf = pd.DataFrame(rows, columns=["jeda_menit", "rho", "n"])
            hasil[s]["ccf"][arah] = ccf.round(4).to_dict("list")
            ax.plot(ccf["jeda_menit"], ccf["rho"], color=ARAH_WARNA[arah], linewidth=2, label=ARAH_NAMA[arah])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvspan(5, 20, color=MUTED, alpha=0.15, linewidth=0)
        ax.set_title(f"{nama}: korelasi silang hujan -> residual arus")
        ax.set_ylabel("Spearman rho")
        if col == 0:
            ax.legend(fontsize=8, loc="upper left")

        # kejadian awal hujan
        jam = pd.read_csv(DATA / f"cuaca_{s}.csv")
        # resolusi efektif: nilai hujan > 0 yang identik beberapa jam berturut-turut (blok 3 jam)
        hh = jam["hujan"]
        run = (hh != hh.shift()).cumsum()
        panjang = jam[hh > 0].groupby(run[hh > 0]).size()
        hasil[s]["jam_hujan"] = int((hh > 0).sum())
        hasil[s]["blok_identik"] = {int(k): int(v) for k, v in panjang.value_counts().sort_index().items()}
        hasil[s]["jam_dalam_blok_3"] = int(panjang[panjang >= 3].sum())
        jam["t"] = pd.to_datetime(jam["waktu"]).dt.tz_localize(TZ)
        h = jam.set_index("t")["hujan"]
        # hujan pada stempel t = jumlah (t-1j, t]; awal jam hujan = t-1j
        onset = h[(h >= AMBANG) & (h.shift(1) < 0.1)].index - pd.Timedelta("1h")
        ax = axes[1, col]
        for t0 in onset:
            for arah in ARAH:
                a = res[(res["arah"] == arah) & (res["waktu"] >= t0 - pd.Timedelta("60min"))
                        & (res["waktu"] <= t0 + pd.Timedelta("120min"))]
                a = a[a["res"].notna()]
                if len(a) < 6:
                    continue
                x = (a["waktu"] - t0).dt.total_seconds() / 60
                sebelum = a.loc[x < 0, "res"]
                sesudah = a.loc[(x >= 0) & (x < 60), "res"]
                hasil[s]["kejadian"].append({
                    "awal_jam_hujan": str(t0), "arah": arah, "hujan_mm": float(h[t0 + pd.Timedelta('1h')]),
                    "bin_sebelum": int(len(sebelum)), "bin_jam_hujan": int(len(sesudah)),
                    "res_sebelum": float(sebelum.mean()) if len(sebelum) else None,
                    "res_jam_hujan": float(sesudah.mean()) if len(sesudah) else None})
                # putus garis di celah rekaman agar tidak tampak menyambung
                seg = (x.diff() > 5).cumsum()
                for _, g in a.assign(x=x).groupby(seg.to_numpy()):
                    ax.plot(g["x"], g["res"], color=ARAH_WARNA[arah], linewidth=1.2, marker="o", markersize=2, alpha=0.8)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.axvline(60, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
        ax.axhline(0, color=INK2, linewidth=0.8)
        ne = len({k["awal_jam_hujan"] for k in hasil[s]["kejadian"]})
        hasil[s]["n_onset_total"], hasil[s]["n_onset_terekam"] = len(onset), ne
        ax.set_title(f"{nama}: {ne} dari {len(onset)} awal hujan terekam kamera")
        ax.set_xlabel("Menit dari awal jam hujan (garis putus: akhir jam pertama)")
        ax.set_ylabel("Residual arus (kend/jam)")
    for ax in axes[0]:
        ax.set_xlabel("Jeda (menit); positif = hujan lebih dulu. Area abu-abu: 5-20 menit")
    simpan(fig, "c05_jeda_hujan")
    (HASIL / "jeda_cuaca.json").write_text(json.dumps(hasil, indent=1, default=str), encoding="utf-8")
    for s in SUMBER:
        for arah in ARAH:
            c = pd.DataFrame(hasil[s]["ccf"][arah])
            best = c.loc[c["rho"].idxmin()]
            print(s, arah, "rho jeda 0:", c.loc[c.jeda_menit == 0, "rho"].iloc[0], "| paling negatif:", best.to_dict(),
                  "| jeda 5-20:", c[(c.jeda_menit >= 5) & (c.jeda_menit <= 20)]["rho"].round(3).tolist())
        print(pd.DataFrame(hasil[s]["kejadian"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
