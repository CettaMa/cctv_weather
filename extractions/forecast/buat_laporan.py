"""Susun forecast/laporan_forecast.html (satu file, gambar tertanam) dari forecast/hasil dan forecast/gambar.

    ..\\.venv\\Scripts\\python.exe -m forecast.buat_laporan
"""
import base64
import json

import pandas as pd

from .common import DATA, F, GAMBAR, HASIL, HORIZONS, MAX_FILL, TRAIN_END, VAL_END
from .cuaca_vs_arus import SUMBER_NAMA, VAR as VAR_NAMA
from .gaya import NAMA, keluarga

R = json.loads((DATA / "ringkasan_deret.json").read_text(encoding="utf-8"))
EV = json.loads((HASIL / "evaluasi.json").read_text(encoding="utf-8"))
CU = json.loads((HASIL / "cuaca_vs_arus.json").read_text(encoding="utf-8"))
TU = json.loads((HASIL / "tuning_gbdt.json").read_text(encoding="utf-8"))
M = pd.read_csv(HASIL / "metrik_ringkas.csv")
KH = pd.read_csv(HASIL / "metrik_kelompok_horizon.csv")
SM = pd.read_csv(HASIL / "metrik_siang_malam.csv")
AR = pd.read_csv(HASIL / "metrik_arah.csv")
KOR = pd.read_csv(HASIL / "korelasi_cuaca.csv")
HK = pd.read_csv(HASIL / "hujan_vs_kering.csv")
KF = pd.read_csv(HASIL / "kontribusi_fitur.csv", index_col=0)["mean_abs_kontribusi"]
BASE = ["persistence", "seasonal_naive", "profil_historis"]
DEEP = ["lstm", "nhits", "tft", "tide", "patchtst", "itransformer"]


def num(x, d=0):
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tabel"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(name, caption, narrow=False):
    data = base64.b64encode((GAMBAR / f"{name}.png").read_bytes()).decode()
    cls = ' class="kecil"' if narrow else ""
    return f'<figure{cls}><img src="data:image/png;base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def label(m):
    fam = keluarga(m)
    s = m.split("_", 1)[1].replace("_", " ") if fam in ("lightgbm", "xgboost") else ""
    return NAMA.get(fam, m) + (f" <span class='abu'>({s})</span>" if s else "")


uji = M[M.tahap == "uji"].set_index("model")
val = M[M.tahap == "validasi"].set_index("model")
urut = uji.sort_values("MAE").index
best = EV["terbaik_uji"]
deep_ada = [d for d in DEEP if d in uji.index]
best_deep = min(deep_ada, key=lambda d: uji.loc[d, "MAE"]) if deep_ada else None
lgb_real = "lightgbm_cuaca_asal"
abl = CU["ablasi"]
a1 = abl["lightgbm_cuaca_asal_vs_lightgbm_tanpa_cuaca"]
a2 = abl["lightgbm_cuaca_sempurna_vs_lightgbm_tanpa_cuaca"]
hj = abl["mae_saat_hujan"]
kor_flow = KOR[KOR.respon == "flow"]
sig = KOR[KOR.q_bh < 0.05]

hasil_tabel = table(
    ["Model", "Skenario cuaca", "MAE uji", "RMSE uji", "WAPE uji", "Skill vs profil", "MAE validasi"],
    [[label(m), uji.loc[m, "skenario"], num(uji.loc[m, "MAE"]), num(uji.loc[m, "RMSE"]), num(uji.loc[m, "WAPE"], 1) + "%",
      num(uji.loc[m, "skill_vs_profil_pct"], 1) + "%", num(val.loc[m, "MAE"])] for m in urut])

pilih = ["profil_historis", "seasonal_naive", lgb_real, "lightgbm_cuaca_sempurna"] + ([best_deep] if best_deep else []) + ["ensemble"]
kelompok = table(["Model"] + list(KH["kelompok_h"].unique()),
                 [[label(m)] + [num(KH[(KH.model == m) & (KH.kelompok_h == k)]["MAE"].iloc[0]) for k in KH["kelompok_h"].unique()]
                  for m in pilih])
siang = table(["Model"] + list(SM["periode"].unique()) + list(AR["arah"].unique()),
              [[label(m)] + [num(SM[(SM.model == m) & (SM.periode == k)]["MAE"].iloc[0]) for k in SM["periode"].unique()]
               + [num(AR[(AR.model == m) & (AR.arah == k)]["MAE"].iloc[0]) for k in AR["arah"].unique()] for m in pilih])

smp_uji = EV["smp"].get("uji", {})
smp_tab = table(["Model (target SMP/jam)", "MAE uji", "WAPE uji"],
                [[label(m) if m not in BASE else NAMA[m], num(v["MAE"]), num(v["WAPE"], 1) + "%"]
                 for m, v in sorted(smp_uji.items(), key=lambda kv: kv[1]["MAE"])])

deep_varian = table(["Model deep", "MAE validasi input 12 jam", "MAE validasi input 24 jam", "Dipilih"],
                    [[NAMA[f], num(v["mae_validasi"].get(f"{f}_in144", float("nan"))),
                      num(v["mae_validasi"].get(f"{f}_in288", float("nan"))), v["dipilih"].split("_in")[1] + " langkah"]
                     for f, v in EV["varian_deep"].items()]) if EV["varian_deep"] else "<p>Model deep belum dijalankan.</p>"

split = table(["Bagian", "Periode (WIB)", "Bin 5 menit teramati per arah", "Baris evaluasi (asal x horizon, 2 arah)"], [
    ["Latih", f"10 Nov sampai {TRAIN_END:%d %b} 00.00", num(R["menjauhi"]["teramati_per_bagian"]["latih"]), "-"],
    ["Validasi", f"{TRAIN_END:%d %b} sampai {VAL_END:%d %b} 00.00", num(R["menjauhi"]["teramati_per_bagian"]["validasi"]),
     num(EV["baris"]["validasi"])],
    ["Uji", f"{VAL_END:%d %b} sampai 26 Nov 11.00", num(R["menjauhi"]["teramati_per_bagian"]["uji"]), num(EV["baris"]["uji"])],
])

cv = table(["Fold", "Asal evaluasi", "Baris latih", "Baris evaluasi"],
           [[i + 1, f"{pd.Timestamp(TU['fold_awal'][i]):%d %b} - {pd.Timestamp(TU['fold_awal'][i + 1]):%d %b}",
             num(r[0]), num(r[1])] for i, r in enumerate(TU["fold_baris"])])
cv_mae = table(["Model", "MAE CV (rata-rata 4 fold)"],
               [[NAMA.get(k, k), num(v)] for k, v in sorted(TU["cv_mae"].items(), key=lambda kv: kv[1])])

fitur = table(["Kelompok", "Fitur", "Keterangan"], [
    ["Nilai terakhir", "<code>y0, smp0, lag1-lag12, rm3/rm6/rm12, n_obs12, panjang_rekam</code>",
     "Hanya data sampai titik asal. Lag kosong (kamera mati) dibiarkan NaN; LightGBM menanganinya."],
    ["Musiman", "<code>kemarin, minggu_lalu, profil_target, profil_asal, anomali0</code>",
     "Nilai pada jam yang sama kemarin/minggu lalu; profil median jam x jenis hari dari data sebelum cutoff."],
    ["Arah lawan", "<code>lawan0, lawan_rm6</code>", "Arus arah lain di titik asal."],
    ["Komposisi dan kondisi", "<code>porsi_motor_rm12, okupansi_rm6, berhenti_rm6</code>", "Rata-rata bergulir 30-60 menit."],
    ["Kalender", "<code>jam_target, jam_sin, jam_cos, dow_target, akhir_pekan, jam_asal, h_menit</code>",
     "Diketahui di muka. Satu model untuk semua horizon; horizon jadi fitur."],
    ["Cuaca saat asal", "<code>hujan_asal, hujan_1j_asal, hujan_3j_asal, suhu_asal, rh_asal</code>", "Skenario cuaca_asal dan cuaca_sempurna."],
    ["Cuaca saat target", "<code>hujan_target, hujan_1j_target, suhu_target, rh_target, awan_target, angin_target</code>",
     "Hanya skenario cuaca_sempurna: cuaca aktual dipakai seolah ramalan cuaca tanpa galat (batas atas manfaat)."],
])

kor_tab = table(["Variabel cuaca", "rho menjauhi (CI 95%)", "rho mendekati (CI 95%)", "q (BH) terkecil"],
                [[VAR_NAMA[v], *[f"{num(r.rho, 2)} ({num(r.ci_lo, 2)} s.d. {num(r.ci_hi, 2)})" for r in
                       kor_flow[kor_flow.cuaca == v].set_index('arah').loc[["menjauhi", "mendekati"]].itertuples()],
                  num(kor_flow[kor_flow.cuaca == v].q_bh.min(), 2)] for v in kor_flow.cuaca.unique()])

hk = table(["Arah", "Respon", "Jam hujan / kering", "Median residual hujan", "Median residual kering", "p Mann-Whitney"],
           [[r.arah, r.respon, f"{r.jam_hujan} / {r.jam_kering}", num(r.median_res_hujan, 3 if abs(r.median_res_hujan) < 5 else 0),
             num(r.median_res_kering, 3 if abs(r.median_res_kering) < 5 else 0), num(r.p_mannwhitney, 2)] for r in HK.itertuples()])

abl_tab = table(["Perbandingan (set uji)", "MAE dengan cuaca", "MAE tanpa cuaca", "Diebold-Mariano", "p"], [
    ["LightGBM cuaca saat asal vs tanpa cuaca", num(a1["mae_a"]), num(a1["mae_b"]), num(a1["dm_stat"], 2), num(a1["p"], 3)],
    ["LightGBM cuaca sempurna vs tanpa cuaca", num(a2["mae_a"]), num(a2["mae_b"]), num(a2["dm_stat"], 2), num(a2["p"], 3)],
])
hujan_tab = table(["Model", f"MAE saat target hujan >= 0,5 mm/jam ({hj['n']} baris)"],
                  [[label(m) if m not in BASE else NAMA[m], num(v)] for m, v in hj.items() if m != "n"])

dm = EV["diebold_mariano"]
dm_txt = "uji Diebold-Mariano " + "; ".join(
    f"terhadap {k.split('_vs_')[1].replace('_', ' ')} DM {num(v['stat'], 2)}, p {num(v['p'], 4)}" for k, v in dm.items())
gain = 100 * (1 - uji.loc[lgb_real, "MAE"] / uji.loc["profil_historis", "MAE"])
deep_val_rank = val.loc[deep_ada, "MAE"].sort_values() if deep_ada else None
gbdt_val_best = val.loc[[m for m in val.index if m.startswith(("lightgbm", "xgboost"))], "MAE"].min()
deep_txt = (f"<b>Peringkat model tidak stabil antara validasi dan uji.</b> Model deep terbaik di uji adalah "
            f"{NAMA[best_deep]} (MAE {num(uji.loc[best_deep, 'MAE'])}), tetapi di validasi MAE-nya "
            f"{num(val.loc[best_deep, 'MAE'])}, lebih buruk daripada GBDT terbaik ({num(gbdt_val_best)}). "
            f"Selisih antar model terbaik di uji hanya beberapa persen, sedangkan set uji hanya 4 hari. Yang konsisten "
            f"di kedua set: LightGBM/XGBoost lebih baik daripada profil historis, dan LSTM, N-HiTS, TiDE, TFT tidak "
            f"lebih baik daripada GBDT."
            ) if best_deep else "Model deep belum dijalankan."

DH = json.loads((HASIL / "deteksi_hujan.json").read_text(encoding="utf-8"))
_lab = pd.read_csv(F / "label_hujan_video.csv")
DH_LAB = {"n_baris": len(_lab), "salah": int(_lab["catatan"].fillna("").str.contains("salah deteksi").sum())}
DH_K = {(k["awal"][:10], k["arah"]): k["respon"] for k in DH["kejadian"]}
_e = DH["evaluasi_detektor"]
dh_eval = table(["Evaluasi leave-one-day-out (siang)", "Nilai"], [
    ["Menit berlabel / basah", f"{num(_e['menit_berlabel'])} / {num(_e['menit_basah'])} ({_e['hari']} hari)"],
    ["AUC per menit", num(_e["auc_menit"], 2)],
    ["Presisi / recall basah (ambang 0,5)", f"{num(_e['presisi_basah'], 2)} / {num(_e['recall_basah'], 2)}"],
    ["AUC per video (rata-rata peluang)", num(_e["auc_video"], 2)],
    ["Fitur terpenting", ", ".join(_e["fitur_terpenting"])],
])
_nm = {"est_flow_veh_per_hour": "Arus (kend/jam)", "average_active_total": "Kendaraan di ROI",
       "avg_stopped_ratio": "Rasio berhenti", "porsi_motor": "Porsi motor"}
def _f(x):
    return "-" if x is None else num(x, 2 if abs(x) < 10 else 0)


dh_tab = table(["Kejadian", "Arah", "Ukuran", "30 menit sebelum", "Menit 0-10 sesudah", "Menit 10-35 sesudah",
                "Hari kerja lain (median)", "Menit pertama di luar p10-p90"],
               [[k["awal"][:16], k["arah"], _nm[m], _f(r["sebelum_30"]), f"{_f(r['sesudah_0_10'])} ({r['n_0_10']} mnt)",
                 f"{_f(r['sesudah_10_35'])} ({r['n_10_35']} mnt)", _f(r["hari_lain_median"]),
                 "-" if r["menit_keluar_rentang"] is None else num(r["menit_keluar_rentang"])]
                for k in DH["kejadian"] for m, r in k["respon"].items()])
JD = json.loads((HASIL / "jeda_cuaca.json").read_text(encoding="utf-8"))
JD_MAX = max(abs(r) for s in JD.values() for c in s["ccf"].values() for r in c["rho"])
CS = json.loads((DATA / "cuaca_sumber.json").read_text(encoding="utf-8"))
SN = SUMBER_NAMA[CS["sumber"]]
if CS["sumber"] == "owm":
    cuaca_txt = (f"OpenWeatherMap One Call {CS.get('versi', '4.0')} per jam (hujan, suhu, kelembapan, awan, angin) "
                 f"di titik CCTV {num(CS['lat'], 7)} / {num(CS['lon'], 7)}, diubah ke 5 menit. "
                 f"{CS['jam_hilang']} jam tidak tersedia dari API (19 Nov 00.00-05.00 WIB) dan diisi interpolasi.")
else:
    cuaca_txt = (f"Open-Meteo historical-forecast per jam di {num(CS['lat'], 7)} / {num(CS['lon'], 7)}, diubah ke 5 menit.")
pb = CS.get("pembanding")
if pb:
    cuaca_txt += (f" <b>Sebagai pembanding</b>, data {SUMBER_NAMA[pb['sumber']]} di titik yang sama sangat berbeda: "
                  f"korelasi hujan per jam hanya {num(pb['korelasi_hujan'], 2)}, kelembapan {num(pb['korelasi_rh'], 2)}, "
                  f"suhu {num(pb['korelasi_suhu'], 2)}; jam hujan >= 0,5 mm/jam {pb['jam_hujan_ini']} vs {pb['jam_hujan_lain']}, "
                  f"total hujan {num(pb['total_hujan_ini_mm'], 1)} vs {num(pb['total_hujan_lain_mm'], 1)} mm. "
                  f"Kesimpulan tentang hujan karena itu sangat bergantung pada sumber cuaca.")
h_list = ", ".join(str(h * 5) for h in HORIZONS[:4]) + ", ..., 360"

html = f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecasting Arus Buahbatu</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; background:#f9f9f7; color:#0b0b0b; font:15px/1.55 system-ui,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:40px 0 8px; border-top:1px solid #e1e0d9; padding-top:20px; }}
h3 {{ font-size:16px; margin:24px 0 6px; }} .sub {{ color:#52514e; margin:0 0 16px; }} .abu {{ color:#52514e; }}
figure {{ margin:14px 0; background:#fcfcfb; border:1px solid #e1e0d9; border-radius:8px; padding:10px; }}
figure img {{ width:100%; height:auto; display:block; }} figure.kecil {{ max-width:640px; }}
figcaption {{ color:#52514e; font-size:13px; margin-top:6px; }}
.tabel {{ overflow-x:auto; margin:12px 0; }}
table {{ border-collapse:collapse; width:100%; background:#fcfcfb; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ border:1px solid #e1e0d9; padding:6px 9px; text-align:left; vertical-align:top; }} th {{ background:#f0efec; }}
code {{ background:#f0efec; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
li {{ margin-bottom:6px; }}
.kotak {{ background:#fcfcfb; border:1px solid #e1e0d9; border-left:4px solid #2a78d6; border-radius:6px; padding:4px 18px; }}
</style></head><body><main>
<h1>Forecasting arus lalu lintas Terusan Buahbatu</h1>
<p class="sub">Target: arus kendaraan/jam per arah, resolusi 5 menit, horizon {h_list} menit. Data 10-26 November 2026, cuaca {SN}.</p>

<h2>1. Kesimpulan</h2>
<div class="kotak"><ul>
<li><b>Rekomendasi: LightGBM.</b> Tanpa ramalan cuaca, MAE uji {num(uji.loc[lgb_real, 'MAE'])} kend/jam dengan cuaca saat asal dan {num(uji.loc['lightgbm_tanpa_cuaca', 'MAE'])} tanpa cuaca, sekitar {num(gain, 0)}% lebih baik daripada profil historis ({num(uji.loc['profil_historis', 'MAE'])}). Bila ramalan cuaca per jam tersedia, cuaca pada waktu target menurunkan MAE sampai {num(uji.loc['lightgbm_cuaca_sempurna', 'MAE'])} (batas atas; MAE validasi {num(val.loc['lightgbm_cuaca_sempurna', 'MAE'])}, termasuk terbaik). LightGBM konsisten di validasi dan uji, dan dilatih dalam hitungan detik. Angka terendah di uji dicapai {label(best)} (MAE {num(uji.loc[best, 'MAE'])}; {dm_txt}).</li>
<li><b>Keunggulan atas profil historis kecil (sekitar 10%).</b> Dengan 17 hari data, pola jam x jenis hari sudah menjelaskan sebagian besar variasi; data terbaru paling membantu pada horizon 1-3 jam. Persistence (nilai 5 menit terakhir) paling buruk karena arus 5 menit sangat berisik.</li>
<li>{deep_txt} Data latih hanya {num(R['menjauhi']['teramati_per_bagian']['latih'])} bin teramati per arah.</li>
<li><b>Hubungan cuaca dan arus lemah dan belum meyakinkan.</b> Korelasi residual arus dengan hujan {num(kor_flow[(kor_flow.cuaca == 'hujan') & (kor_flow.arah == 'menjauhi')].rho.iloc[0], 2)} (menjauhi) dan {num(kor_flow[(kor_flow.cuaca == 'hujan') & (kor_flow.arah == 'mendekati')].rho.iloc[0], 2)} (mendekati); {'tidak ada' if sig.empty else str(len(sig))} dari {len(KOR)} uji korelasi yang signifikan setelah koreksi uji berganda. Hanya {CU['jam_hujan_ge_0.5']['menjauhi']} jam terekam yang hujannya >= 0,5 mm/jam.</li>
<li><b>Jeda hujan ke arus (bagian 5.4)</b>: dari video ditemukan 2 awal hujan yang terekam. Hujan deras mendadak (16 Nov) berdampak dalam 0-3 menit; hujan yang menguat bertahap (19 Nov) berdampak sekitar 10-25 menit kemudian. Dua kejadian belum cukup untuk menyimpulkan pola umum.</li>
<li><b>Cuaca hanya membantu bila cuaca di waktu target diketahui.</b> Fitur cuaca saat asal menurunkan MAE LightGBM {num(a1['mae_b'] - a1['mae_a'], 1)} (p {num(a1['p'], 3)}); cuaca aktual pada waktu target menurunkan {num(a2['mae_b'] - a2['mae_a'], 1)} (p {num(a2['p'], 3)}). Saat target hujan, MAE turun dari {num(hj['lightgbm_tanpa_cuaca'])} ke {num(hj['lightgbm_cuaca_sempurna'])}, tetapi itu dari sedikit kejadian hujan. Fitur cuaca menyumbang {num(abl['porsi_kontribusi_cuaca_pct'], 1)}% kontribusi prediksi.</li>
</ul></div>
{fig('f02_mae_model', 'MAE set uji, semua horizon dan kedua arah. Abu-abu: baseline.', narrow=True)}

<h2>2. Penyiapan data</h2>
<p>Sumber: <code>eda/data/dataset_menit_berflag.csv</code>. Langkah di <code>forecast/siapkan_data.py</code>:</p>
<ul>
<li><b>Tipe data</b>: <code>menit</code> ke datetime zona Asia/Jakarta, kolom flag dan <code>layak</code> ke boolean, <code>arah</code> dan kondisi ke kategori, angka ke float32.</li>
<li><b>Hanya menit layak</b> (dari EDA). Bin 5 menit dianggap teramati bila berisi minimal 3 menit layak; nilainya rata-rata berbobot durasi sampel.</li>
<li><b>Grid reguler</b>: {num(R['bin_total_per_arah'])} bin per arah, {num(R['menjauhi']['teramati'])} teramati ({num(100 * R['menjauhi']['teramati'] / R['bin_total_per_arah'], 1)}%).</li>
<li><b>Imputasi terbatas</b>: celah sampai {MAX_FILL * 5} menit yang diapit dua bin teramati diisi profil musiman + interpolasi linear residual ({num(R['menjauhi']['diimputasi'])} bin per arah). {num(R['menjauhi']['kosong'])} bin dalam celah panjang tetap kosong dan diberi penanda <code>panjang_celah</code> / <code>langkah_sejak_obs</code>. Nilai isian hanya dipakai sebagai input; target dan evaluasi selalu memakai bin teramati.</li>
<li><b>Model deep</b> butuh input tanpa lubang: celah panjang diisi profil musiman dari data sebelum cutoff, kolom <code>isian</code> memberi tahu model mana yang asli, dan <code>available_mask</code> membuat loss tidak dihitung pada isian.</li>
<li><b>Cuaca</b>: {cuaca_txt}</li>
</ul>

<h3>2.1 Split kronologis dan cross validation</h3>
{split}
<p>Batas dipilih agar bin teramati sekitar 73/11/16%; 21-22 dan 24-26 November banyak menit tidak layak. Titik asal ramalan hanya bin teramati (kamera sedang merekam), dan target harus teramati.</p>
<p>Tuning memakai CV rolling-origin dengan jendela mengembang di dalam bagian latih saja, supaya validasi tetap bersih untuk memilih varian dan ensemble. Baris latih tiap fold hanya yang targetnya sebelum awal fold (purged).</p>
{cv}
{cv_mae}

<h2>3. Feature engineering</h2>
{fitur}

<h2>4. Model dan hasil</h2>
<ul>
<li><b>Baseline</b>: persistence, seasonal naive (nilai kemarin pada jam yang sama, atau profil bila kosong), profil historis.</li>
<li><b>LightGBM dan XGBoost</b>: satu model global direct multi-horizon untuk kedua arah, tuning Optuna (40 percobaan LightGBM, 20 XGBoost) pada CV, hiperparameter sama untuk tiga skenario cuaca.</li>
<li><b>Deep (neuralforecast, GPU)</b>: LSTM, N-HiTS, TFT, TiDE dengan kovariat kalender + cuaca dan arus arah lawan; PatchTST dan iTransformer tanpa kovariat (tidak didukung). Keluaran 72 langkah (6 jam), loss MAE, early stopping, input 12 atau 24 jam dipilih di validasi. Arus diskalakan global (dibagi 1.000) karena scaler per-jendela bawaan neuralforecast menghasilkan ramalan yang meledak pada jendela yang hampir seluruhnya isian; ramalan negatif dipotong ke 0.</li>
<li><b>Ensemble</b>: rata-rata 3 model dengan MAE validasi terbaik: {', '.join(label(m) for m in EV['ensemble_dari'])}.</li>
</ul>
{hasil_tabel}
<p class="sub">Skill = penurunan MAE terhadap profil historis. Semua model dinilai pada {num(EV['baris']['uji'])} baris yang sama ({num(EV['asal_uji'])} titik asal).</p>
{fig('f01_mae_horizon', 'MAE per horizon di set uji. Tiap titik dari sekitar 100-280 baris, jadi garisnya bergelombang. Persistence tidak digambar (MAE naik sampai sekitar 1.600).')}
{kelompok}
{siang}
{fig('f03_contoh_ramalan', 'Contoh ramalan 6 jam dari satu titik asal per arah. Titik: arus teramati (2 jam sebelum asal sampai 6 jam setelahnya).')}

<h3>4.1 Varian model deep</h3>
{deep_varian}

<h3>4.2 Target kedua: SMP/jam</h3>
{smp_tab}

<h2>5. Cuaca dan arus</h2>
<p>Arus per jam (minimal 6 bin teramati) dikurangi profil jam x jenis hari, supaya hujan sore tidak tertukar dengan pola sore. Nilai p memakai n efektif (koreksi autokorelasi lag 1) dan q memakai koreksi Benjamini-Hochberg untuk {len(KOR)} uji (6 variabel cuaca x 4 respon x 2 arah).</p>
{fig('c01_hujan_dan_residual', 'Atas: hujan per jam. Bawah: residual arus pada jam yang terekam. Banyak hujan jatuh saat kamera tidak merekam.')}
{fig('c03_korelasi_cuaca', 'Korelasi Spearman residual arus per jam dengan cuaca. Garis: CI 95% dengan n efektif.', narrow=True)}
{kor_tab}
{fig('c02_residual_per_intensitas', 'Residual arus menurut intensitas hujan. n = jumlah jam.')}
<h3>5.1 Hujan vs kering pada jam yang sama</h3>
{hk}
<h3>5.2 Manfaat cuaca untuk ramalan</h3>
{abl_tab}
{hujan_tab}
{fig('c04_kontribusi_fitur', 'Kontribusi fitur LightGBM skenario cuaca sempurna di set uji (rata-rata nilai absolut pred_contrib).', narrow=True)}

<h3>5.3 Apakah arus berubah 5-20 menit setelah cuaca berubah?</h3>
<p><b>Belum bisa dijawab dengan data yang ada.</b> Hasil pemeriksaan (<code>forecast/jeda_cuaca.py</code>):</p>
<ul>
<li><b>Korelasi silang</b> residual arus 5 menit dengan hujan yang digeser -60 sampai +180 menit: semua |rho| di bawah {num(JD_MAX, 2)}, tanpa puncak di jeda 5-20 menit pada kedua sumber.</li>
<li><b>Awal hujan terlalu sedikit</b>: {JD['owm']['n_onset_total']} awal hujan (>= 0,5 mm/jam setelah jam kering) menurut OpenWeatherMap dan {JD['openmeteo']['n_onset_total']} menurut Open-Meteo selama 18 hari. Kamera merekam sebagian sekitar {JD['owm']['n_onset_terekam']} dan {JD['openmeteo']['n_onset_terekam']} di antaranya, sering hanya beberapa bin 5 menit. Arah perubahan arus tidak konsisten antar kejadian.</li>
<li><b>Resolusi cuaca terlalu kasar.</b> Kedua sumber per jam, jadi menit mulainya hujan tidak diketahui. Hujan historis OpenWeatherMap bahkan sebagian berupa blok 3 jam: {JD['owm']['jam_dalam_blok_3']} dari {JD['owm']['jam_hujan']} jam hujan bernilai identik tiga jam berturut-turut (dimulai di jam kelipatan 3), artinya resolusi efektifnya 3 jam ({JD['openmeteo']['jam_dalam_blok_3']} dari {JD['openmeteo']['jam_hujan']} pada Open-Meteo).</li>
<li><b>Waktu kedua sumber juga tidak cocok dengan video.</b> Pada 19 November, video menunjukkan jalan sudah basah sejak sekitar 13.20 WIB. Open-Meteo mencatat hujan pukul 12-14, sedangkan OpenWeatherMap baru mulai 14.00 lalu 3,02 mm/jam selama tiga jam.</li>
<li><b>Jam data lalu lintas sendiri belum pasti</b>: overlay video sekitar 11-13 menit di belakang nama file. Ketidakpastian ini seukuran jeda yang ingin diuji.</li>
</ul>
{fig('c05_jeda_hujan', 'Atas: korelasi silang hujan dengan residual arus (area abu-abu: jeda 5-20 menit). Bawah: residual arus di sekitar awal jam hujan, garis putus di celah rekaman.')}
{fig('c06_video_19nov', 'Video 19 November 13.22-14.22 WIB (nama file), satu frame tiap 8 menit. Jalan sudah basah di frame pertama.')}
<p>Karena itu waktu mulai hujan diambil dari video CCTV sendiri (bagian 5.4).</p>

<h3>5.4 Detektor hujan dari video dan arus di sekitar awal hujan</h3>
<p><b>Cara kerja</b> (<code>forecast/hujan_video.py</code>, <code>forecast/deteksi_hujan.py</code>): satu frame tiap 6 detik; median frame per menit menghapus kendaraan sehingga tersisa latar jalan. Dari latar itu dihitung kecerahan, tekstur, saturasi, dan kilap (pantulan) jalan dibanding area lain, serta ketajaman frame. LightGBM memetakan fitur itu ke peluang jalan basah. Jam yang dipakai sama dengan data lalu lintas (nama file + posisi frame), sehingga selisih jam overlay tidak memengaruhi jeda.</p>
<p><b>Label</b>: {DH_LAB['n_baris']} rentang menit di {DH['evaluasi_detektor']['video_berlabel']} video siang hari, dari pemeriksaan visual frame yang saya (Claude) lakukan, bukan dari sensor atau anotator manusia (<code>forecast/label_hujan_video.csv</code>). Survei mencakup 96 video yang dekat jam hujan menurut salah satu sumber cuaca, lima frame per video; perubahan dari kering ke basah diperiksa per 1-2 menit pada resolusi penuh.</p>
{dh_eval}
<ul>
<li><b>Detektor menemukan kedua awal hujan sebelum keduanya dilabel</b> (19 Nov di putaran pertama, 16 Nov di putaran kedua), dengan selisih 2 menit lebih awal (16 Nov) dan 5 menit lebih lambat (19 Nov) dibanding pengamatan visual.</li>
<li><b>Ada {DH_LAB['salah']} deteksi palsu</b>, semuanya karena perubahan cahaya: bayangan matahari pagi dan senja. Ketiganya lalu dimasukkan sebagai label (contoh sulit). Angka evaluasi di atas termasuk label itu, jadi agak optimistis untuk data baru.</li>
<li>Hanya siang (06.00-17.59). Pada malam hari kamera berpindah mode dan jalan basah belum bisa dilabel dengan yakin.</li>
<li><b>Hanya 2 awal hujan yang terekam</b> di seluruh data. Awal hujan lain (misalnya 20, 24, 25 Nov) jatuh di celah antar rekaman.</li>
</ul>
{fig('c08_detektor_kejadian', 'Peluang jalan basah per menit pada dua video yang memuat awal hujan. Garis oranye: awal hujan menurut pemeriksaan visual.')}
{dh_tab}
{fig('c07_arus_sekitar_awal_hujan', 'Arus, kendaraan di ROI, rasio berhenti, dan porsi motor per menit di sekitar dua awal hujan. Video 16 Nov berakhir sekitar 5 menit setelah hujan mulai; data 19 Nov dimulai 15 menit sebelumnya.')}
<p><b>Apa yang terlihat (2 kejadian, deskriptif):</b></p>
<ul>
<li><b>16 Nov (Senin, hujan deras mendadak)</b>: perubahan terjadi <b>segera</b>, 0-3 menit setelah hujan, bukan 5-20 menit. Arus menjauhi turun dari rata-rata {num(DH_K[('2026-11-16','menjauhi')]['est_flow_veh_per_hour']['sebelum_30'])} (30 menit sebelum) ke {num(DH_K[('2026-11-16','menjauhi')]['est_flow_veh_per_hour']['sesudah_0_10'])} kend/jam pada menit 0-5, rasio berhenti melonjak ke sekitar 0,4, dan porsi motor turun tajam. Ini cocok dengan pengendara motor menepi atau berteduh, tetapi rekaman berhenti 5 menit kemudian, jadi kelanjutannya tidak terlihat.</li>
<li><b>19 Nov (Kamis, hujan bertahap)</b>: tidak ada perubahan langsung. Kendaraan di ROI arah mendekati turun mulai sekitar menit ke-{num(DH_K[('2026-11-19','mendekati')]['average_active_total']['menit_keluar_rentang'] or 0)}, dan arusnya keluar dari rentang normal sekitar menit ke-{num(DH_K[('2026-11-19','mendekati')]['est_flow_veh_per_hour']['menit_keluar_rentang'] or 0)}. Arah menjauhi justru naik dan berfluktuasi. Hujan pada kejadian ini makin deras beberapa menit setelah awalnya (jalan basah penuh sekitar menit ke-6).</li>
<li><b>Kesimpulan sementara</b>: jeda yang Anda amati mungkin bergantung pada cara hujan datang. Hujan deras mendadak berdampak segera, sedangkan hujan yang menguat bertahap berdampak setelah 10-25 menit pada kejadian ini. Dua kejadian belum cukup untuk menyimpulkan pola umum atau menguji signifikansi.</li>
<li><b>Hati-hati dengan angka saat hujan deras</b>: tetes air di lensa dan jarak pandang yang turun dapat menurunkan deteksi YOLO dan memutus track. Sebagian penurunan arus dan lonjakan rasio berhenti bisa berasal dari detektor, bukan lalu lintas. Akurasi deteksi saat hujan belum divalidasi.</li>
</ul>
<p>Untuk menguji jeda secara statistik dibutuhkan lebih banyak awal hujan yang terekam, idealnya rekaman menerus selama musim hujan. Detektor ini bisa dijalankan pada rekaman baru untuk menemukannya otomatis, dengan pemeriksaan visual singkat untuk menyaring deteksi palsu akibat cahaya.</p>

<h2>6. Batasan</h2>
<ul>
<li><b>Data pendek dan berlubang</b>: 17 hari, rekaman sekitar 1 jam lalu celah 1-14 jam. Set uji hanya {num(R['menjauhi']['teramati_per_bagian']['uji'])} bin per arah (4 hari kerja), jadi peringkat model yang selisihnya kecil bisa berubah dengan data lain.</li>
<li><b>Ramalan 6 jam di sini hanya untuk jam yang kebetulan terekam.</b> Galatnya mewakili jam-jam itu, tidak seluruh hari.</li>
<li><b>Skenario cuaca sempurna</b> memakai cuaca aktual sebagai ramalan; dalam operasi nyata galat ramalan cuaca akan mengurangi manfaatnya.</li>
<li><b>Cuaca dari model grid</b>, bukan stasiun di lokasi. Hujan lokal yang singkat bisa terlewat.</li>
<li><b>Target sendiri belum tervalidasi penuh</b>: arus baru dibandingkan dengan hitungan manual pada satu klip 60 detik, malam hari belum divalidasi, dan sejak 21 November porsi menit layak turun. Residual arus juga cenderung negatif sejak 19 November (lihat gambar residual), mungkin karena mutu rekaman.</li>
<li>Jam pada overlay video sekitar 13 menit di belakang nama file; analisis cuaca per jam tidak terlalu terpengaruh, tetapi waktu tepatnya belum pasti.</li>
</ul>

<h2>7. Menjalankan ulang</h2>
<p>Dari folder <code>extractions</code> dengan <code>..\\.venv\\Scripts\\python.exe -m</code>: <code>forecast.cuaca</code>, <code>forecast.siapkan_data</code>, <code>forecast.model_gbdt</code>, <code>forecast.model_deep</code>, <code>forecast.evaluasi</code>, <code>forecast.cuaca_vs_arus</code>, <code>forecast.buat_laporan</code>. Tabel hasil ada di <code>forecast/hasil/</code>.</p>
</main></body></html>
"""
(F / "laporan_forecast.html").write_text(html, encoding="utf-8")
print("laporan:", F / "laporan_forecast.html", f"{len(html) / 1e6:.1f} MB")
