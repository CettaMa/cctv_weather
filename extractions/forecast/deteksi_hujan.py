"""Detektor jalan basah siang hari dari fitur video per menit, dan analisis arus di sekitar awal hujan.

Jalankan dari folder `extractions` setelah `forecast.hujan_video`:
    ..\\.venv\\Scripts\\python.exe -m forecast.deteksi_hujan

1. Label: rentang menit kering/basah dari pemeriksaan visual frame video
   (`forecast/label_hujan_video.csv`; dibuat dengan melihat frame, bukan sensor).
2. LightGBM pada fitur latar per menit + jam, dinilai dengan leave-one-day-out (semua menit dari satu
   tanggal ditahan bersama). Hanya siang (06.00-17.59): di malam hari kamera beralih ke mode malam
   dan jalan basah belum bisa dilabel dengan yakin.
3. Model akhir memberi peluang basah untuk semua menit siang; awal hujan = minimal 8 menit kering
   lalu minimal 8 menit basah di dalam satu video.
4. Untuk tiap awal hujan: arus, kendaraan di ROI, rasio berhenti, dan porsi motor per menit dari
   60 menit sebelum sampai 45 menit sesudah, dibanding menit yang sama pada hari kerja lain.
Jam yang dipakai adalah jam nama file, sama untuk video dan data lalu lintas.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import DATA, F, HASIL, TZ
from .gaya import ARAH_NAMA, ARAH_WARNA, INK2, MUTED, plt, simpan

FITUR = ["terang_jalan", "terang_lain", "rasio_terang", "std_jalan", "tekstur_jalan", "tekstur_lain",
         "rasio_tekstur", "sat_jalan", "kilap_jalan", "tajam_frame", "jam_f"]
MIN_RUN = 8
SEED = 7


def muat_fitur() -> pd.DataFrame:
    parts = []
    for f in sorted((DATA / "hujan_video").glob("*.csv")):
        d = pd.read_csv(f)
        d["video"] = f.stem
        parts.append(d)
    v = pd.concat(parts, ignore_index=True)     # video tanpa frame terbaca menghasilkan CSV kosong
    v[FITUR[:-1]] = v[FITUR[:-1]].astype(float)
    v["t"] = pd.to_datetime(v["menit"], utc=True).dt.tz_convert(TZ)
    v["jam_f"] = v["t"].dt.hour + v["t"].dt.minute / 60
    v["tanggal"] = v["t"].dt.date.astype(str)
    return v.sort_values(["video", "t"]).reset_index(drop=True)


def beri_label(v: pd.DataFrame) -> pd.Series:
    lab = pd.Series(np.nan, index=v.index)
    for r in pd.read_csv(F / "label_hujan_video.csv").itertuples():
        m = v["video"] == r.video
        sel = m & (v["t"].dt.strftime("%H:%M").between(r.dari, r.sampai))
        lab[sel] = 1.0 if r.label == "basah" else 0.0
    return lab


def runs(b: np.ndarray) -> list[tuple[int, int, bool]]:
    out, s = [], 0
    for i in range(1, len(b) + 1):
        if i == len(b) or b[i] != b[s]:
            out.append((s, i, bool(b[s])))
            s = i
    return out


def main() -> None:
    v = muat_fitur()
    siang = (v["t"].dt.hour >= 6) & (v["t"].dt.hour < 18)
    v["label"] = beri_label(v)
    lab = v[siang & v["label"].notna()].copy()
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 20,
              "feature_fraction": 0.8, "verbosity": -1, "seed": SEED}
    oof = pd.Series(np.nan, index=lab.index)
    for tgl in lab["tanggal"].unique():
        te = lab["tanggal"] == tgl
        tr = lab[~te]
        if tr["label"].nunique() < 2:
            continue
        b = lgb.train(params, lgb.Dataset(tr[FITUR], tr["label"]), 200)
        oof[te] = b.predict(lab.loc[te, FITUR])
    ok = oof.notna()
    y, p = lab.loc[ok, "label"], oof[ok]
    # per video: label mayoritas vs prediksi rata-rata (cara detektor dipakai)
    per_video = lab.loc[ok].assign(p=p).groupby("video").agg(label=("label", "mean"), p=("p", "mean"))
    evaluasi = {
        "menit_berlabel": int(len(lab)), "menit_basah": int(lab["label"].sum()), "hari": int(lab["tanggal"].nunique()),
        "video_berlabel": int(lab["video"].nunique()),
        "auc_menit": float(roc_auc_score(y, p)), "akurasi_menit_0.5": float(((p >= 0.5) == y).mean()),
        "recall_basah": float(((p >= 0.5) & (y == 1)).sum() / (y == 1).sum()),
        "presisi_basah": float(((p >= 0.5) & (y == 1)).sum() / max((p >= 0.5).sum(), 1)),
        "auc_video": float(roc_auc_score((per_video["label"] > 0.5).astype(int), per_video["p"])),
    }
    final = lgb.train(params, lgb.Dataset(lab[FITUR], lab["label"]), 200)
    imp = pd.Series(final.feature_importance("gain"), index=FITUR).sort_values(ascending=False)
    evaluasi["fitur_terpenting"] = imp.head(5).round(0).to_dict()

    # peluang basah semua menit siang, dihaluskan 5 menit per video
    vs = v[siang].copy()
    vs["p"] = final.predict(vs[FITUR])
    vs["p5"] = vs.groupby("video")["p"].transform(lambda s: s.rolling(5, center=True, min_periods=2).median())
    onset = []
    for vid, g in vs.groupby("video"):
        b = (g["p5"] >= 0.5).to_numpy()
        rr = runs(b)
        for (s0, e0, w0), (s1, e1, w1) in zip(rr, rr[1:]):
            if not w0 and w1 and e0 - s0 >= MIN_RUN and e1 - s1 >= MIN_RUN:
                onset.append({"video": vid, "awal": g["t"].iloc[s1].isoformat(), "kering_menit": e0 - s0,
                              "basah_menit": e1 - s1, "p_sebelum": float(g["p"].iloc[s0:e0].mean()),
                              "p_sesudah": float(g["p"].iloc[s1:e1].mean())})
    vs[["video", "menit", "p", "p5"]].to_csv(HASIL / "peluang_basah_per_menit.csv", index=False)
    verif = pd.read_csv(F / "awal_hujan_terverifikasi.csv")
    verif["t0"] = pd.to_datetime(verif["awal"]).dt.tz_localize(TZ)
    for o in onset:     # cocokkan deteksi dengan awal hujan yang terverifikasi visual (selisih <= 15 menit)
        d = (verif["t0"] - pd.Timestamp(o["awal"])).abs()
        o["terverifikasi"] = bool((d <= pd.Timedelta("15min")).any())
        o["selisih_menit_vs_visual"] = float(d.min().total_seconds() / 60) if o["terverifikasi"] else None

    # ---- arus per menit di sekitar awal hujan
    m = pd.read_csv(F.parent / "eda" / "data" / "dataset_menit_berflag.csv")
    m["t"] = pd.to_datetime(m["menit"], utc=True).dt.tz_convert(TZ)
    m["porsi_motor"] = m["average_active_motor"] / m["average_active_total"].where(m["average_active_total"] > 0)
    respon = {"est_flow_veh_per_hour": "Arus (kend/jam, jendela 60 dtk)", "average_active_total": "Kendaraan di ROI",
              "avg_stopped_ratio": "Rasio berhenti", "porsi_motor": "Porsi motor"}
    kejadian = []
    fig, grid = plt.subplots(len(respon), len(verif), figsize=(5.5 * len(verif), 9.5), sharex=True, squeeze=False)
    for j, o in enumerate(verif.itertuples()):
        axes = grid[:, j]
        t0 = o.t0
        jam0 = t0.hour * 60 + t0.minute
        for arah in ("menjauhi", "mendekati"):
            a = m[m["arah"] == arah].copy()
            a["dt"] = (a["t"] - t0).dt.total_seconds() / 60
            ev = a[(a["dt"] >= -60) & (a["dt"] <= 45)].sort_values("dt")
            # pembanding: menit jam yang sama pada hari kerja lain (bukan tanggal kejadian)
            a["mnt"] = a["t"].dt.hour * 60 + a["t"].dt.minute
            lain = a[(a["t"].dt.date != t0.date()) & (a["t"].dt.dayofweek < 5) & a["layak"]
                     & (a["mnt"] >= jam0 - 60) & (a["mnt"] <= jam0 + 45)]
            ring = {}
            for k in respon:
                s0 = ev.loc[(ev["dt"] >= -30) & (ev["dt"] < 0), k]
                s1 = ev.loc[(ev["dt"] >= 0) & (ev["dt"] < 10), k]
                s2 = ev.loc[(ev["dt"] >= 10) & (ev["dt"] < 35), k]
                ring[k] = {"sebelum_30": float(s0.mean()), "sesudah_0_10": float(s1.mean()),
                           "n_0_10": int(s1.notna().sum()),
                           "sesudah_10_35": float(s2.mean()) if s2.notna().any() else None,
                           "n_10_35": int(s2.notna().sum()), "hari_lain_median": float(lain[k].median())}
                # menit pertama setelah awal hujan di mana rata-rata 5 menit keluar dari rentang p10-p90 hari lain
                lo, hi = lain[k].quantile(0.1), lain[k].quantile(0.9)
                r5 = ev.set_index("dt")[k].rolling(5, min_periods=3).mean()
                keluar = r5[(r5.index >= 0) & ((r5 < lo) | (r5 > hi))]
                ring[k]["menit_keluar_rentang"] = float(keluar.index[0]) if len(keluar) else None
            kejadian.append({"awal": str(t0), "arah": arah, "menit_ada": int(len(ev)),
                             "menit_layak": int(ev["layak"].sum()), "respon": ring})
            for ax, (k, nama) in zip(axes, respon.items()):
                ax.plot(ev["dt"], ev[k].rolling(3, center=True, min_periods=1).mean(), color=ARAH_WARNA[arah],
                        linewidth=1.8, label=ARAH_NAMA[arah] if k == "est_flow_veh_per_hour" else None)
                lo, hi = lain[k].quantile(0.1), lain[k].quantile(0.9)
                ax.axhspan(lo, hi, color=ARAH_WARNA[arah], alpha=0.08, linewidth=0)
                if j == 0:
                    ax.set_ylabel(nama, fontsize=8)
        for ax in axes:
            ax.axvline(0, color=INK2, linewidth=1)
            ax.axvspan(5, 20, color=MUTED, alpha=0.12, linewidth=0)
        axes[0].set_title(f"Awal hujan {t0:%a %d %b %H.%M}", fontsize=10)
        axes[-1].set_xlabel("Menit dari awal hujan (jam nama file)")
    grid[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Pita: p10-p90 hari kerja lain pada jam yang sama. Abu-abu: 5-20 menit setelah awal hujan. "
                 "Garis: rata-rata 3 menit.", fontsize=8, color=INK2, y=0.995)
    simpan(fig, "c07_arus_sekitar_awal_hujan")

    # ---- grafik peluang basah untuk video kejadian
    if len(verif):
        fig, axs = plt.subplots(1, len(verif), figsize=(5.5 * len(verif), 2.8), sharey=True, squeeze=False)
        for ax, o in zip(axs[0], verif.itertuples()):
            g = vs[vs["video"] == o.video]
            dt = (g["t"] - o.t0).dt.total_seconds() / 60
            ax.plot(dt, g["p"], color=MUTED, linewidth=1, label="per menit")
            ax.plot(dt, g["p5"], color="#2a78d6", linewidth=2, label="median 5 menit")
            ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
            ax.axvline(0, color="#eb6834", linewidth=1.5, label="awal hujan (visual)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{o.t0:%d %b}, video {o.video[17:]}", fontsize=10)
            ax.set_xlabel("Menit dari awal hujan")
        axs[0, 0].set_ylabel("Peluang jalan basah")
        axs[0, 0].legend(fontsize=8, loc="upper left")
        simpan(fig, "c08_detektor_kejadian")

    hasil = {"evaluasi_detektor": evaluasi, "awal_hujan_terdeteksi": onset, "kejadian": kejadian,
             "catatan": "label visual dari frame video; siang saja"}
    (HASIL / "deteksi_hujan.json").write_text(json.dumps(hasil, indent=1, default=str), encoding="utf-8")
    print(json.dumps(hasil, indent=1, default=str))


if __name__ == "__main__":
    main()
