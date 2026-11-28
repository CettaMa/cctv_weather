"""Hubungan cuaca dan arus lalu lintas.

Jalankan dari folder `extractions` (setelah model_gbdt dan evaluasi):
    ..\\.venv\\Scripts\\python.exe -m forecast.cuaca_vs_arus

1. Arus per jam (rata-rata bin 5 menit teramati, minimal 6 bin) dikurangi profil jam x jenis hari,
   supaya pola harian (hujan Bandung sering sore hari) tidak terbaca sebagai efek cuaca.
2. Spearman residual vs variabel cuaca, plus korelasi parsial (dikontrol jam dan jenis hari via
   residual, serta suhu/kelembapan untuk hujan). Data per jam berautokorelasi: nilai p dihitung juga
   dengan n efektif (koreksi lag-1).
3. Hujan vs kering (Mann-Whitney pada residual) untuk arus, porsi motor, rasio berhenti, okupansi.
4. Ablasi model: LightGBM tanpa cuaca / cuaca saat asal / cuaca sempurna, uji Diebold-Mariano,
   dan kontribusi fitur (pred_contrib) model cuaca sempurna di set uji.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from .common import ARAH, DATA, HASIL, VAL_END, load_cuaca_5menit, load_deret

SUMBER_NAMA = {"owm": "OpenWeatherMap", "openmeteo": "Open-Meteo"}
from .evaluasi import diebold_mariano
from .fitur import FITUR_CUACA_ASAL, FITUR_CUACA_TARGET, SKENARIO, tabel
from .gaya import ARAH_NAMA, ARAH_WARNA, INK2, MUTED, SURFACE, plt, simpan
from .siapkan_data import profil

VAR = {"hujan": "Hujan (mm/jam)", "hujan_3j": "Hujan 3 jam terakhir (mm)", "suhu": "Suhu (°C)",
       "rh": "Kelembapan (%)", "awan": "Tutupan awan (%)", "angin": "Angin (km/jam)"}
RESPON = {"flow": "Arus", "porsi_motor": "Porsi motor", "rasio_berhenti": "Rasio berhenti", "okupansi": "Okupansi ROI"}


def per_jam(deret: pd.DataFrame, cuaca: pd.DataFrame) -> pd.DataFrame:
    deret = deret.copy()
    deret["porsi_motor"] = deret["motor"] / deret["aktif"].where(deret["aktif"] > 0)
    # residual 5 menit terhadap profil halus (median jam x jenis hari dari seluruh data, interpolasi antar-jam)
    for arah in ARAH:
        m = deret["arah"] == arah
        sub = deret[m].reset_index(drop=True)
        for k in RESPON:
            deret.loc[m, f"res_{k}"] = (sub[k] - profil(sub, k, pd.Timestamp.max.tz_localize("UTC"))).to_numpy()
    d = deret[deret["teramati"]].copy()
    d["jam_bin"] = d["waktu"].dt.floor("h")
    agg = {k: (k, "mean") for k in RESPON} | {f"res_{k}": (f"res_{k}", "mean") for k in RESPON}
    g = d.groupby(["arah", "jam_bin"]).agg(n=("flow", "size"), **agg).reset_index()
    g = g[g["n"] >= 6]
    c = cuaca.set_index("waktu")
    ch = c.groupby(c.index.floor("h")).agg({"hujan": "mean", "hujan_3j": "last", "suhu": "mean", "rh": "mean",
                                            "awan": "mean", "angin": "mean"})
    g = g.join(ch, on="jam_bin")
    g["jam"] = g["jam_bin"].dt.hour
    g["akhir_pekan"] = (g["jam_bin"].dt.dayofweek >= 5).astype(int)
    return g


def n_efektif(x: np.ndarray, y: np.ndarray) -> float:
    r1 = lambda v: pd.Series(v).autocorr(1) if len(v) > 3 else 0.0
    a, b = np.nan_to_num(r1(x)), np.nan_to_num(r1(y))
    return len(x) * (1 - a * b) / (1 + a * b)


def spearman(x, y) -> dict:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    rho, p = stats.spearmanr(x, y)
    ne = max(n_efektif(stats.rankdata(x), stats.rankdata(y)), 4)
    t = rho * np.sqrt((ne - 2) / max(1 - rho ** 2, 1e-9))
    p_eff = 2 * stats.t.sf(abs(t), ne - 2)
    z, se = np.arctanh(rho), 1 / np.sqrt(ne - 3)
    return {"n": int(len(x)), "n_efektif": round(float(ne), 1), "rho": float(rho), "p": float(p),
            "p_n_efektif": float(p_eff), "ci_lo": float(np.tanh(z - 1.96 * se)), "ci_hi": float(np.tanh(z + 1.96 * se))}


def parsial(df: pd.DataFrame, x: str, y: str, kontrol: list) -> dict:
    """Spearman parsial: rank, regresikan kontrol, korelasikan residual."""
    d = df[[x, y, *kontrol]].dropna().rank()
    A = np.c_[np.ones(len(d)), d[kontrol].to_numpy()]
    rx = d[x] - A @ np.linalg.lstsq(A, d[x], rcond=None)[0]
    ry = d[y] - A @ np.linalg.lstsq(A, d[y], rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return {"n": int(len(d)), "rho_parsial": float(r), "p": float(p)}


def main() -> None:
    deret, cuaca = load_deret(), load_cuaca_5menit()
    g = per_jam(deret, cuaca)
    g.to_csv(HASIL / "arus_cuaca_per_jam.csv", index=False)
    kor, beda, hasil = [], [], {}
    for arah in ARAH:
        a = g[g["arah"] == arah]
        for v in VAR:
            for k in RESPON:
                kor.append({"arah": arah, "cuaca": v, "respon": k, **spearman(a[v].to_numpy(), a[f"res_{k}"].to_numpy())})
        hasil[f"parsial_hujan_{arah}"] = parsial(a, "hujan", "res_flow", ["suhu", "rh"])
        hujan = a["hujan"] >= 0.5
        for k in RESPON:
            x1, x0 = a.loc[hujan, f"res_{k}"].dropna(), a.loc[~hujan, f"res_{k}"].dropna()
            u = stats.mannwhitneyu(x1, x0) if len(x1) >= 3 else None
            beda.append({"arah": arah, "respon": k, "jam_hujan": len(x1), "jam_kering": len(x0),
                         "median_res_hujan": float(x1.median()) if len(x1) else np.nan,
                         "median_res_kering": float(x0.median()), "p_mannwhitney": float(u.pvalue) if u else np.nan,
                         "median_level_hujan": float(a.loc[hujan, k].median()) if len(x1) else np.nan})
    kor = pd.DataFrame(kor)
    from statsmodels.stats.multitest import multipletests
    kor["q_bh"] = multipletests(kor["p_n_efektif"], method="fdr_bh")[1]   # koreksi uji berganda (48 uji)
    kor = kor.round(4)
    beda = pd.DataFrame(beda).round(4)
    kor.to_csv(HASIL / "korelasi_cuaca.csv", index=False)
    beda.to_csv(HASIL / "hujan_vs_kering.csv", index=False)

    # -------- ablasi model dan kontribusi fitur
    uji = pd.read_parquet(HASIL / "pred_uji_gabungan.parquet")
    abl = {}
    for a_, b_ in [("lightgbm_cuaca_asal", "lightgbm_tanpa_cuaca"), ("lightgbm_cuaca_sempurna", "lightgbm_tanpa_cuaca")]:
        s, p = diebold_mariano(uji[a_] - uji["y"], uji[b_] - uji["y"], uji["waktu_asal"])
        abl[f"{a_}_vs_{b_}"] = {"mae_a": float(np.mean(np.abs(uji[a_] - uji["y"]))),
                                "mae_b": float(np.mean(np.abs(uji[b_] - uji["y"]))), "dm_stat": s, "p": p}
    hujan_target = uji.merge(cuaca.assign(waktu_target=cuaca["waktu"].dt.tz_localize(None))[["waktu_target", "hujan"]],
                             on="waktu_target", how="left")
    basah = hujan_target["hujan"] >= 0.5
    abl["mae_saat_hujan"] = {c: float(np.mean(np.abs(hujan_target.loc[basah, c] - hujan_target.loc[basah, "y"])))
                             for c in ["lightgbm_tanpa_cuaca", "lightgbm_cuaca_asal", "lightgbm_cuaca_sempurna",
                                       "profil_historis"]}
    abl["mae_saat_hujan"]["n"] = int(basah.sum())
    tab = tabel(deret, cuaca, cutoff=VAL_END)
    ev = tab[tab["waktu_asal"] >= VAL_END]
    fit = SKENARIO["cuaca_sempurna"]
    b = lgb.Booster(model_file=str(HASIL / "model" / "lightgbm_cuaca_sempurna.txt"))
    contrib = np.abs(b.predict(ev[fit], pred_contrib=True)[:, :-1]).mean(axis=0)
    kontribusi = pd.Series(contrib, index=fit).sort_values(ascending=False)
    kontribusi.rename("mean_abs_kontribusi").to_csv(HASIL / "kontribusi_fitur.csv")
    cu_cols = FITUR_CUACA_ASAL + FITUR_CUACA_TARGET
    abl["porsi_kontribusi_cuaca_pct"] = float(100 * kontribusi[cu_cols].sum() / kontribusi.sum())
    hasil["ablasi"] = abl
    hasil["jam_teramati"] = {arah: int((g["arah"] == arah).sum()) for arah in ARAH}
    hasil["jam_hujan_ge_0.5"] = {arah: int(((g["arah"] == arah) & (g["hujan"] >= 0.5)).sum()) for arah in ARAH}
    (HASIL / "cuaca_vs_arus.json").write_text(json.dumps(hasil, indent=1), encoding="utf-8")

    # -------- grafik c01: hujan dan residual arus per jam (dua panel, satu sumbu masing-masing)
    c = cuaca.set_index("waktu")["hujan"]
    ch = c.groupby(c.index.floor("h")).mean()
    ch = ch[(ch.index >= g["jam_bin"].min()) & (ch.index <= g["jam_bin"].max())]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a1.bar(ch.index, ch.values, width=1 / 24, color="#2a78d6")
    a1.set_ylabel("mm/jam")
    sumber = json.loads((DATA / "cuaca_sumber.json").read_text(encoding="utf-8"))["sumber"]
    a1.set_title(f"Hujan per jam ({SUMBER_NAMA[sumber]})")
    for arah in ARAH:
        a = g[g["arah"] == arah]
        a2.scatter(a["jam_bin"], a["res_flow"], s=6, color=ARAH_WARNA[arah], label=ARAH_NAMA[arah])
    a2.axhline(0, color=INK2, linewidth=0.8)
    a2.set_ylabel("kendaraan/jam")
    a2.set_title("Residual arus terhadap profil jam x jenis hari (jam yang terekam)")
    a2.legend(fontsize=8, ncol=2, loc="lower left", markerscale=2)
    simpan(fig, "c01_hujan_dan_residual")

    # -------- c02: residual arus menurut intensitas hujan
    g["kelas"] = pd.cut(g["hujan"], [-0.01, 0.1, 0.5, 2, np.inf], labels=["kering", "<0.5", "0.5-2", ">2"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, arah in zip(axes, ARAH):
        a = g[g["arah"] == arah]
        data = [a.loc[a["kelas"] == k, "res_flow"].dropna() for k in a["kelas"].cat.categories]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": SURFACE, "linewidth": 2})
        for p in bp["boxes"]:
            p.set(facecolor=ARAH_WARNA[arah], edgecolor=ARAH_WARNA[arah])
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=MUTED)
        ax.set_xticks(range(1, 5), [f"{k}\n(n={len(d)})" for k, d in zip(a["kelas"].cat.categories, data)])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(ARAH_NAMA[arah])
        ax.set_xlabel("Hujan (mm/jam)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Residual arus (kendaraan/jam)")
    simpan(fig, "c02_residual_per_intensitas")

    # -------- c03: korelasi Spearman residual arus vs cuaca, dengan CI n efektif
    k = kor[kor["respon"] == "flow"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    vs = list(VAR)
    for i, arah in enumerate(ARAH):
        s = k[k["arah"] == arah].set_index("cuaca").reindex(vs)
        y = np.arange(len(vs)) + (i - 0.5) * 0.3
        ax.errorbar(s["rho"], y, xerr=[s["rho"] - s["ci_lo"], s["ci_hi"] - s["rho"]], fmt="o",
                    color=ARAH_WARNA[arah], markersize=5, capsize=0, linewidth=1.5, label=ARAH_NAMA[arah])
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(vs)), [VAR[v] for v in vs])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho residual arus (CI 95%, n efektif)")
    ax.set_title("Korelasi cuaca dengan residual arus per jam")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    simpan(fig, "c03_korelasi_cuaca")

    # -------- c04: kontribusi fitur LightGBM (cuaca sempurna)
    top = kontribusi.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(range(len(top)), top.values, height=0.65,
            color=["#eb6834" if f in cu_cols else "#2a78d6" for f in top.index])
    ax.set_yticks(range(len(top)), top.index)
    ax.set_xlabel("Rata-rata |kontribusi| ke prediksi (kendaraan/jam)")
    ax.set_title("15 fitur terpenting LightGBM (oranye = cuaca)")
    ax.grid(axis="y", visible=False)
    simpan(fig, "c04_kontribusi_fitur")

    print(kor[kor["respon"] == "flow"].to_string(index=False))
    print(beda.to_string(index=False))
    print(json.dumps(hasil, indent=1))


if __name__ == "__main__":
    main()
