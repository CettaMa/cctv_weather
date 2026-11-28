"""Gabung semua ramalan, pilih varian/ensemble di validasi, nilai di uji, buat grafik.

Jalankan dari folder `extractions` (setelah model_gbdt dan model_deep):
    ..\\.venv\\Scripts\\python.exe -m forecast.evaluasi
"""
import json

import numpy as np
import pandas as pd

from .common import HASIL, TZ, metrik
from .gaya import ARAH_NAMA, GARIS, INK2, MUTED, NAMA, WARNA, keluarga, plt, simpan

KUNCI = ["arah", "waktu_asal", "waktu_target", "h", "y"]
BASELINE = ["persistence", "seasonal_naive", "profil_historis"]


def naive(s: pd.Series) -> pd.Series:
    return s.dt.tz_convert(TZ).dt.tz_localize(None) if s.dt.tz is not None else s


def gabung(nama: str) -> pd.DataFrame:
    p = pd.read_parquet(HASIL / f"pred_{nama}.parquet")
    p["waktu_asal"], p["waktu_target"] = naive(p["waktu_asal"]), naive(p["waktu_target"])
    f = HASIL / f"pred_deep_{nama}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        p = p.merge(d, on=["arah", "waktu_asal", "waktu_target"], how="left", validate="1:1")
    return p


def skenario(kolom: str) -> str:
    if kolom in BASELINE:
        return "-"
    if kolom == "ensemble":
        return "campuran"
    if kolom.startswith(("lightgbm", "xgboost")):
        return kolom.split("_", 1)[1]
    if kolom.startswith(("patchtst", "itransformer")):
        return "tanpa kovariat"
    return "cuaca_sempurna"


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, waktu: pd.Series, lag: int = 12) -> tuple[float, float]:
    """DM pada selisih galat absolut yang dirata-rata per titik asal; varians Newey-West."""
    from scipy import stats
    d = pd.Series(np.abs(e1) - np.abs(e2)).groupby(waktu.to_numpy()).mean().sort_index().to_numpy()
    n = len(d)
    dc = d - d.mean()
    v = np.sum(dc * dc) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def main() -> None:
    val, uji = gabung("validasi"), gabung("uji")
    model_cols = [c for c in val.columns if c not in KUNCI]
    # varian panjang input model deep: pilih di validasi
    pilihan = {}
    for fam in sorted({c.split("_in")[0] for c in model_cols if "_in" in c}):
        var = [c for c in model_cols if c.startswith(fam + "_in")]
        mae = {c: np.mean(np.abs(val[c] - val["y"])) for c in var}
        best = min(mae, key=mae.get)
        pilihan[fam] = {"dipilih": best, "mae_validasi": {k: round(float(v), 1) for k, v in mae.items()}}
        for df in (val, uji):
            df[fam] = df[best]
            df.drop(columns=var, inplace=True)
    model_cols = [c for c in val.columns if c not in KUNCI]
    kandidat = [c for c in model_cols if c not in BASELINE]
    mae_val = {c: float(np.mean(np.abs(val[c] - val["y"]))) for c in kandidat}
    top = sorted(mae_val, key=mae_val.get)[:3]
    for df in (val, uji):
        df["ensemble"] = df[top].mean(axis=1)
    model_cols.append("ensemble")
    for df in (val, uji):
        assert df[model_cols].notna().all().all(), df[model_cols].isna().sum()

    rows = []
    for nama, df in (("validasi", val), ("uji", uji)):
        ref = {b: np.mean(np.abs(df[b] - df["y"])) for b in ("profil_historis", "seasonal_naive")}
        for c in model_cols:
            m = metrik(df["y"].to_numpy(), df[c].to_numpy())
            rows.append({"tahap": nama, "model": c, "skenario": skenario(c), **m,
                         "skill_vs_profil_pct": 100 * (1 - m["MAE"] / ref["profil_historis"]),
                         "skill_vs_seasonal_pct": 100 * (1 - m["MAE"] / ref["seasonal_naive"])})
    ringkas = pd.DataFrame(rows).round(2)
    ringkas.to_csv(HASIL / "metrik_ringkas.csv", index=False)

    def per(df, by):
        out = []
        for key, g in df.groupby(by):
            for c in model_cols:
                out.append({by if isinstance(by, str) else "grup": key, "model": c,
                            **metrik(g["y"].to_numpy(), g[c].to_numpy())})
        return pd.DataFrame(out).round(2)

    uji["h_menit"] = uji["h"] * 5
    hz = per(uji, "h_menit")
    hz.to_csv(HASIL / "metrik_horizon.csv", index=False)
    per(uji, "arah").to_csv(HASIL / "metrik_arah.csv", index=False)
    jam = uji["waktu_target"].dt.hour
    uji["periode"] = np.where((jam >= 6) & (jam < 18), "siang (06-18)", "malam (18-06)")
    per(uji, "periode").to_csv(HASIL / "metrik_siang_malam.csv", index=False)
    uji["kelompok_h"] = pd.cut(uji["h_menit"], [0, 60, 180, 360], labels=["5-60 menit", "75-180 menit", "195-360 menit"])
    per(uji, "kelompok_h").to_csv(HASIL / "metrik_kelompok_horizon.csv", index=False)

    terbaik = ringkas[(ringkas.tahap == "uji")].sort_values("MAE")
    best = terbaik[~terbaik.model.isin(BASELINE)].iloc[0]["model"]
    dm = {}
    for lawan in ("profil_historis", "seasonal_naive"):
        s, p = diebold_mariano(uji[best] - uji["y"], uji[lawan] - uji["y"], uji["waktu_asal"])
        dm[f"{best}_vs_{lawan}"] = {"stat": round(s, 2), "p": round(p, 4)}
    smp = {}
    for nama in ("validasi", "uji"):
        f = HASIL / f"pred_smp_{nama}.parquet"
        if f.exists():
            s = pd.read_parquet(f)
            smp[nama] = {c: metrik(s["y"].to_numpy(), s[c].to_numpy())
                         for c in s.columns if c not in KUNCI}
    info = {"varian_deep": pilihan, "ensemble_dari": top, "mae_validasi": mae_val, "terbaik_uji": best,
            "diebold_mariano": dm, "smp": smp,
            "baris": {"validasi": len(val), "uji": len(uji)},
            "asal_uji": int(uji["waktu_asal"].nunique())}
    (HASIL / "evaluasi.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    uji.to_parquet(HASIL / "pred_uji_gabungan.parquet", index=False)

    # ---------------- grafik 1: MAE per horizon, dua panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panel = [   # persistence tidak digambar (MAE sampai 1.600 menenggelamkan perbedaan lain); ada di tabel
        ("Baseline vs model tabular", ["seasonal_naive", "profil_historis", "lightgbm_cuaca_asal",
                                        "xgboost_cuaca_asal", "ensemble"]),
        ("Model deep vs LightGBM (cuaca sempurna)", ["lightgbm_cuaca_sempurna", "lstm", "nhits", "tft", "tide",
                                                     "patchtst", "itransformer"]),
    ]
    for ax, (judul, cols) in zip(axes, panel):
        for c in [c for c in cols if c in hz.model.unique()]:
            s = hz[hz.model == c].set_index("h_menit")["MAE"]
            fam = keluarga(c)
            ax.plot(s.index, s.values, color=WARNA.get(fam, INK2), linestyle=GARIS.get(fam, "-"), linewidth=2,
                    marker="o", markersize=3, label=NAMA.get(fam, c) + ("" if fam == c or fam in ("ensemble",)
                                                                       else f" ({skenario(c).replace('_', ' ')})"
                                                                       if fam in ("lightgbm", "xgboost") else ""))
        ax.set_title(judul)
        ax.set_xlabel("Horizon (menit)")
        ax.set_xticks([5, 60, 120, 180, 240, 300, 360])
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axes[0].set_ylabel("MAE (kendaraan/jam)")
    axes[0].set_ylim(0)
    simpan(fig, "f01_mae_horizon")

    # ---------------- grafik 2: MAE uji per model
    t = terbaik.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 0.8))
    warna = [MUTED if m in BASELINE else WARNA.get(keluarga(m), INK2) for m in t.model]
    ax.barh(range(len(t)), t.MAE, color=warna, height=0.65)
    lab = [f"{NAMA.get(keluarga(m), m)}" + (f" - {s.replace('_', ' ')}" if s not in ("-",) else "")
           for m, s in zip(t.model, t.skenario)]
    ax.set_yticks(range(len(t)), lab)
    for i, v in enumerate(t.MAE):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("MAE uji (kendaraan/jam), semua horizon")
    ax.grid(axis="y", visible=False)
    ax.set_title("Galat rata-rata di set uji (23-26 Nov)")
    simpan(fig, "f02_mae_model")

    # ---------------- grafik 3: contoh ramalan
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    deep = [c for c in ("lstm", "nhits", "tft", "tide", "patchtst", "itransformer") if c in uji]
    deep_best = [min(deep, key=lambda c: mae_val.get(c, np.inf))] if deep else []
    for ax, arah in zip(axes, ("menjauhi", "mendekati")):
        u = uji[uji.arah == arah]
        cnt = u.groupby("waktu_asal").size()
        asal = cnt[cnt == cnt.max()].index[len(cnt[cnt == cnt.max()]) // 2]
        g = u[u.waktu_asal == asal].sort_values("waktu_target")
        akt = val_uji_aktual(arah, asal)
        ax.scatter(akt.index, akt.values, s=6, color=INK2, label="Aktual (teramati)", zorder=3)
        for c in ["profil_historis", "lightgbm_cuaca_asal", *deep_best, "ensemble"]:
            fam = keluarga(c)
            ax.plot(g.waktu_target, g[c], color=WARNA[fam], linestyle=GARIS.get(fam, "-"), linewidth=2,
                    marker="o", markersize=3, label=NAMA[fam])
        ax.axvline(asal, color=MUTED, linewidth=1)
        ax.set_title(f"{ARAH_NAMA[arah]}: asal {asal:%d %b %H:%M}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Arus (kendaraan/jam)")
    axes[0].legend(fontsize=8, loc="lower left")
    simpan(fig, "f03_contoh_ramalan")
    print(ringkas[ringkas.tahap == "uji"].sort_values("MAE").to_string(index=False))
    print(json.dumps({k: info[k] for k in ("varian_deep", "ensemble_dari", "terbaik_uji", "diebold_mariano")}, indent=1))


def val_uji_aktual(arah: str, asal) -> pd.Series:
    from .common import load_deret
    d = load_deret()
    d = d[d.arah == arah]
    t = naive(d["waktu"])
    m = (t >= asal - pd.Timedelta("2h")) & (t <= asal + pd.Timedelta("6h"))
    return pd.Series(d.loc[m, "flow"].to_numpy(), index=t[m]).dropna()


if __name__ == "__main__":
    main()
