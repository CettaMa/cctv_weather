"""EDA dataset menit: gabung dua arah, beri flag mutu, buat grafik dan statistik.

Jalankan dari folder `extractions`, setelah `eda/pindai_mutu.py`:
    ..\\.venv\\Scripts\\python.exe eda\\analisis.py      (butuh pandas dan matplotlib)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

E = Path(__file__).resolve().parent
ROOT = E.parent
DATA, FIG = E / "data", E / "gambar"
FIG.mkdir(exist_ok=True)

DIRS = {"menjauhi": "Menjauhi kamera", "mendekati": "Mendekati kamera"}
COLOR = {"menjauhi": "#2a78d6", "mendekati": "#eb6834"}
CLASS_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASSES = ["motor", "mobil", "bus", "truk"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("biru", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
FPS = 25
# ambang flag mutu
MIN_COVERAGE, MAX_CORRUPT, MAX_PARKED, MIN_SECONDS = 0.90, 0.10, 0.10, 45

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ data
corrupt = pd.read_csv(DATA / "frame_rusak_per_menit.csv", parse_dates=["menit"])
parked = pd.read_csv(DATA / "parkir_per_menit.csv", parse_dates=["menit"])
frames = {}
for key in DIRS:
    m = pd.read_csv(ROOT / f"results_{key}" / "subsession_1min_pkji2.csv", encoding="utf-8-sig")
    m["menit"] = pd.to_datetime(m.one_min_bucket)
    m = m.merge(corrupt, on="menit", how="left").merge(parked, on="menit", how="left")
    m["cakupan"] = (m.sampled_frames / (FPS * 60)).clip(upper=1)
    m["porsi_rusak"] = m.porsi_rusak.fillna(0)
    m["porsi_parkir"] = m.porsi_parkir.fillna(0) if key == "mendekati" else 0.0
    m["flag_cakupan_rendah"] = m.cakupan < MIN_COVERAGE
    m["flag_frame_rusak"] = m.porsi_rusak > MAX_CORRUPT
    m["flag_menit_pendek"] = m.actual_sample_duration_seconds < MIN_SECONDS
    m["flag_tanpa_arus"] = m.est_flow_veh_per_hour.isna()
    m["flag_parkir"] = m.porsi_parkir > MAX_PARKED
    m["layak"] = ~(m.flag_cakupan_rendah | m.flag_frame_rusak | m.flag_menit_pendek | m.flag_tanpa_arus)
    m["jam"] = m.menit.dt.hour
    m["tanggal"] = m.menit.dt.date
    m["akhir_pekan"] = m.menit.dt.dayofweek >= 5
    m["arah"] = key
    frames[key] = m
both = pd.concat(frames.values(), ignore_index=True)

KEEP = ["menit", "arah", "video_file", "layak", "flag_cakupan_rendah", "flag_frame_rusak", "flag_menit_pendek",
        "flag_tanpa_arus", "flag_parkir", "cakupan", "porsi_rusak", "porsi_parkir", "actual_sample_duration_seconds",
        "average_active_motor", "average_active_mobil", "average_active_bus", "average_active_truk",
        "average_active_total", "avg_smp_frame", "avg_roi_occupancy_pct", "avg_pixel_occupancy_pct",
        "est_flow_veh_per_hour", "est_q_smp_per_hour", "avg_dj_flow", "avg_dj_spatial", "avg_dj_effective",
        "max_dj_effective", "avg_moving_vehicles", "avg_slow_vehicles", "avg_stopped_vehicles", "avg_stopped_ratio",
        "pct_time_lancar", "pct_time_padat", "pct_time_macet", "one_min_traffic_condition"]
table = both[KEEP].sort_values(["menit", "arah"])
table.round({c: 4 for c in table.select_dtypes("number")}).to_csv(DATA / "dataset_menit_berflag.csv", index=False)

# ------------------------------------------------------------------ statistik
S = {"ambang": {"cakupan_min": MIN_COVERAGE, "rusak_maks": MAX_CORRUPT, "parkir_maks": MAX_PARKED}}
VARS = {"est_flow_veh_per_hour": "Arus (kend/jam)", "est_q_smp_per_hour": "Arus (SMP/jam)", "avg_dj_effective": "DJ efektif",
        "avg_roi_occupancy_pct": "Okupansi ROI (%)", "average_active_total": "Kendaraan di ROI",
        "avg_stopped_ratio": "Rasio berhenti"}
for key, m in frames.items():
    ok = m[m.layak]
    v = pd.read_csv(ROOT / f"results_{key}" / "video_summary_pkji2.csv", encoding="utf-8-sig")
    span = int((m.menit.max() - m.menit.min()).total_seconds() // 60) + 1
    comp = m[[f"average_active_{c}" for c in CLASSES]].mul(m.sampled_frames, axis=0).sum()
    series = ok.set_index("menit").est_flow_veh_per_hour.sort_index()
    lag = {}
    for k in (1, 15, 60):
        shifted = series.reindex(series.index - pd.Timedelta(minutes=k))
        lag[k] = round(float(np.corrcoef(series[shifted.notna().values], shifted.dropna())[0, 1]), 3)
    S[key] = {
        "video": len(v), "status_video": v.status.value_counts().to_dict(), "menit": len(m), "menit_rentang": span,
        "awal": str(m.menit.min()), "akhir": str(m.menit.max()), "hari": int(m.tanggal.nunique()),
        "layak": int(m.layak.sum()), "flag": {c[5:]: int(m[c].sum()) for c in m.columns if c.startswith("flag_")},
        "deskriptif": ok[list(VARS)].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_dict(),
        "kondisi": ok.one_min_traffic_condition.value_counts().to_dict(),
        "kondisi_semua": m.one_min_traffic_condition.value_counts().to_dict(),
        "komposisi_pct": {c: round(float(100 * comp[f"average_active_{c}"] / comp.sum()), 2) for c in CLASSES},
        "dj_lebih_1": int((ok.avg_dj_effective > 1).sum()),
        "dj_dari_arus_pct": round(float(100 * (ok.avg_dj_flow >= ok.avg_dj_spatial).mean()), 1),
        "emp_high_pct": round(float(100 * ok.dominant_emp_mode.str.startswith("HIGH").mean()), 1),
        "autokorelasi_arus": lag,
        "weekday_weekend_arus": {("akhir_pekan" if k else "hari_kerja"): round(float(g.est_flow_veh_per_hour.median()), 0)
                                 for k, g in ok.groupby("akhir_pekan")},
        "arus_per_jam_median": ok.groupby("jam").est_flow_veh_per_hour.median().round(0).to_dict(),
        "hari_per_jam": ok.groupby("jam").tanggal.nunique().to_dict(),
        "stop_malam": {"dengan_parkir": round(float(ok[ok.flag_parkir].avg_stopped_ratio.mean() or 0), 3),
                       "tanpa_parkir": round(float(ok[~ok.flag_parkir].avg_stopped_ratio.mean()), 3)},
    }
    # arus relatif terhadap median jam dan jenis hari yang sama (dari menit bermutu baik)
    base = ok[(ok.cakupan >= 0.98) & (ok.porsi_rusak == 0)].groupby(["jam", "akhir_pekan"]).est_flow_veh_per_hour.median()
    rel = m[(~m.flag_menit_pendek) & (~m.flag_tanpa_arus)].join(base.rename("acuan"), on=["jam", "akhir_pekan"])
    rel["relatif"] = rel.est_flow_veh_per_hour / rel.acuan
    rel["k_cakupan"] = pd.cut(rel.cakupan, [0, .7, .8, .9, .98, 1.01], right=False,
                              labels=["<70%", "70-80%", "80-90%", "90-98%", ">=98%"])
    rel["k_rusak"] = pd.cut(rel.porsi_rusak, [-1, 0, .1, .3, 1.01], labels=["0", "0-10%", "10-30%", ">30%"])
    for col in ("k_cakupan", "k_rusak"):
        g = rel.groupby(col, observed=False).relatif
        S[key][col] = {str(a): [None if np.isnan(b) else round(float(b), 3), int(n)] for (a, b), n in zip(g.median().items(), g.size())}
# perbandingan di dalam video yang sama: menit bercakupan rendah vs menit utuh
LOW_BINS = ["<80%", "80-90%", "90-98%"]
for key, m in frames.items():
    x = m[~m.flag_menit_pendek & ~m.flag_tanpa_arus & ~m.video_file.str.contains("|", regex=False)]
    x = x.assign(k=pd.cut(x.cakupan, [0, .8, .9, .98, 1.01], right=False, labels=LOW_BINS + ["utuh"]))
    g = x.groupby(["video_file", "k"], observed=True)[["est_flow_veh_per_hour", "average_active_total"]].mean()
    full = g.xs("utuh", level="k")
    out = {}
    for name in LOW_BINS:
        j = g.xs(name, level="k").join(full, lsuffix="_rendah", rsuffix="_utuh", how="inner")
        j = j[(j.est_flow_veh_per_hour_utuh > 200) & (j.average_active_total_utuh > .3)]
        out[name] = {"video": len(j),
                     "arus": round(float((j.est_flow_veh_per_hour_rendah / j.est_flow_veh_per_hour_utuh).median()), 3),
                     "di_roi": round(float((j.average_active_total_rendah / j.average_active_total_utuh).median()), 3)}
    S[key]["dalam_video"] = out
night = frames["mendekati"]
night = night[night.layak & night.jam.isin([22, 23, 0, 1, 2, 3, 4])]
S["parkir_malam"] = {name: {"menit": len(g), "rasio_berhenti": round(float(g.avg_stopped_ratio.mean()), 3),
                            "kendaraan_di_roi": round(float(g.average_active_total.mean()), 3),
                            "okupansi": round(float(g.avg_roi_occupancy_pct.mean()), 2)}
                     for name, g in (("dengan_parkir", night[night.flag_parkir]), ("tanpa_parkir", night[~night.flag_parkir]))}
pair = frames["menjauhi"][frames["menjauhi"].layak].merge(
    frames["mendekati"][frames["mendekati"].layak], on="menit", suffixes=("_a", "_b"))
S["korelasi_antar_arah"] = {k: round(float(pair[f"{k}_a"].corr(pair[f"{k}_b"])), 3)
                            for k in ("est_flow_veh_per_hour", "avg_dj_effective", "avg_roi_occupancy_pct")}
(DATA / "statistik.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")

# ------------------------------------------------------------------ grafik
ref = frames["menjauhi"]
dates = sorted(ref.tanggal.unique())
labels = [pd.Timestamp(d).strftime("%d %b") for d in dates]


def day_hour(m, col, agg):
    return m.pivot_table(index="tanggal", columns="jam", values=col, aggfunc=agg).reindex(index=dates, columns=range(24))


def heat(ax, table, cmap, vmin, vmax, title):
    im = ax.imshow(np.ma.masked_invalid(table.values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(len(dates)), labels)
    ax.set_xticks(np.arange(-.5, 24), minor=True)
    ax.set_yticks(np.arange(-.5, len(dates)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="both", length=0)
    ax.set_title(title)
    ax.set_xlabel("Jam (WIB)")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# 1-2: cakupan dan mutu
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
count = ref.pivot_table(index="tanggal", columns="jam", values="menit", aggfunc="size").reindex(index=dates, columns=range(24))
im = heat(axes[0], count, SEQ, 0, 60, "Menit yang punya data")
fig.colorbar(im, ax=axes[0], shrink=.7, label="menit per jam")
good = day_hour(ref, "layak", "mean") * 100
im = heat(axes[1], good, SEQ, 0, 100, "Porsi menit layak pakai")
fig.colorbar(im, ax=axes[1], shrink=.7, label="%")
save(fig, "01_cakupan")

fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True)
g = ref.groupby("tanggal")
for ax, (series, title) in zip(axes, [
        (g.cakupan.mean() * 100, "Rata-rata porsi frame yang ada (%)"),
        (g.porsi_rusak.mean() * 100, "Rata-rata porsi frame rusak (%)"),
        (g.layak.mean() * 100, "Menit layak pakai (%)")]):
    ax.bar(range(len(dates)), series.reindex(dates), width=.7, color="#2a78d6")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
axes[-1].set_xticks(range(len(dates)), labels, rotation=45, ha="right")
fig.tight_layout()
save(fig, "02_mutu_per_tanggal")

# 3: profil harian
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, (col, title) in zip(axes, [("est_flow_veh_per_hour", "Arus (kendaraan/jam)"),
                                   ("avg_dj_effective", "Derajat kejenuhan efektif"),
                                   ("avg_roi_occupancy_pct", "Okupansi ROI (%)")]):
    for key, m in frames.items():
        q = m[m.layak].groupby("jam")[col].quantile([.25, .5, .75]).unstack().reindex(range(24))
        ax.fill_between(q.index, q[.25], q[.75], color=COLOR[key], alpha=.15, linewidth=0)
        ax.plot(q.index, q[.5], color=COLOR[key], linewidth=2, label=DIRS[key])
    ax.set_title(title)
    ax.set_ylim(0)
axes[0].legend(ncol=2, loc="upper left")
axes[-1].set_xticks(range(0, 24))
axes[-1].set_xlabel("Jam (WIB). Garis = median, pita = kuartil 25-75")
fig.tight_layout()
save(fig, "03_profil_jam")

# 4: hari kerja vs akhir pekan
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (key, m) in zip(axes, frames.items()):
    for weekend, style, name in ((False, "-", "Hari kerja"), (True, (0, (4, 2)), "Akhir pekan")):
        s = m[m.layak & (m.akhir_pekan == weekend)].groupby("jam").est_flow_veh_per_hour.median().reindex(range(24))
        ax.plot(s.index, s, linestyle=style, color=COLOR[key], linewidth=2, marker="o", markersize=3, label=name)
    ax.set_title(DIRS[key])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Jam (WIB)")
    ax.set_ylim(0)
    ax.legend(loc="upper left")
axes[0].set_ylabel("Median arus (kend/jam)")
fig.tight_layout()
save(fig, "04_hari_kerja")

# 5: tanggal x jam
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout="constrained")
vmax = float(both[both.layak].est_flow_veh_per_hour.quantile(.98))
for ax, (key, m) in zip(axes, frames.items()):
    im = heat(ax, day_hour(m[m.layak], "est_flow_veh_per_hour", "median"), SEQ, 0, vmax, DIRS[key])
fig.colorbar(im, ax=axes, shrink=.7, label="median arus (kend/jam)")
save(fig, "05_arus_tanggal_jam")

# 6: komposisi
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
for col_i, (key, m) in enumerate(frames.items()):
    g = m[m.layak].groupby("jam")[[f"average_active_{c}" for c in CLASSES]].mean().reindex(range(24)).fillna(0)
    share = g.div(g.sum(axis=1).replace(0, np.nan), axis=0) * 100
    for row, table in enumerate((g, share)):
        ax, bottom = axes[row, col_i], np.zeros(24)
        for c, color in zip(CLASSES, CLASS_COLOR):
            vals = table[f"average_active_{c}"].fillna(0).values
            ax.bar(range(24), vals, bottom=bottom, color=color, width=.75, edgecolor=SURFACE, linewidth=.6, label=c)
            bottom += vals
        ax.grid(axis="x", visible=False)
    axes[0, col_i].set_title(f"{DIRS[key]}: rata-rata kendaraan di ROI")
    axes[1, col_i].set_title("Komposisi (%)")
    axes[1, col_i].set_xticks(range(0, 24, 3))
    axes[1, col_i].set_xlabel("Jam (WIB)")
top = max(a.get_ylim()[1] for a in axes[0])
for a in axes[0]:
    a.set_ylim(0, top)
axes[0, 0].legend(ncol=4, loc="upper left")
fig.tight_layout()
save(fig, "06_komposisi")

# 7: distribusi
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex="col")
for i, (col, title) in enumerate([("est_flow_veh_per_hour", "Arus (kend/jam)"), ("avg_dj_effective", "DJ efektif"),
                                  ("avg_roi_occupancy_pct", "Okupansi ROI (%)")]):
    bins = np.linspace(0, float(both[both.layak][col].quantile(.995)), 36)
    for r, (key, m) in enumerate(frames.items()):
        x = m[m.layak][col].dropna()
        axes[r, i].hist(x.clip(upper=bins[-1]), bins=bins, color=COLOR[key], edgecolor=SURFACE, linewidth=.5)
        axes[r, i].axvline(x.median(), color=INK, linewidth=1, linestyle=(0, (3, 2)))
        axes[r, i].set_title(f"{title}: {key}" , fontsize=9)
        axes[r, i].grid(axis="x", visible=False)
axes[0, 0].set_ylabel("menit")
axes[1, 0].set_ylabel("menit")
fig.tight_layout()
save(fig, "07_distribusi")

# 8: arus vs okupansi
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, layout="constrained")
xmax = float(both[both.layak].avg_roi_occupancy_pct.quantile(.995))
for ax, (key, m) in zip(axes, frames.items()):
    x = m[m.layak]
    hb = ax.hexbin(x.avg_roi_occupancy_pct, x.est_flow_veh_per_hour, gridsize=35, cmap=SEQ, mincnt=1, bins="log",
                   extent=(0, xmax, 0, float(both[both.layak].est_flow_veh_per_hour.max())), linewidths=.2)
    ax.set_title(DIRS[key])
    ax.set_xlabel("Okupansi ROI (%)")
axes[0].set_ylabel("Arus (kend/jam)")
fig.colorbar(hb, ax=axes, shrink=.8, label="jumlah menit (skala log)")
save(fig, "08_arus_okupansi")

# 9: korelasi Spearman
cols = {"est_flow_veh_per_hour": "arus kend", "est_q_smp_per_hour": "arus SMP", "avg_dj_flow": "DJ arus",
        "avg_dj_spatial": "DJ ruang", "avg_dj_effective": "DJ efektif", "average_active_motor": "motor",
        "average_active_mobil": "mobil", "average_active_truk": "truk", "avg_roi_occupancy_pct": "okupansi ROI",
        "avg_pixel_occupancy_pct": "okupansi piksel", "avg_stopped_ratio": "rasio berhenti", "pct_time_padat": "% padat"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, layout="constrained")
for ax, (key, m) in zip(axes, frames.items()):
    c = m[m.layak][list(cols)].corr(method="spearman")
    im = ax.imshow(c.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols.values(), rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols.values())
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title(DIRS[key])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if abs(c.values[i, j]) > .65 else INK)
fig.colorbar(im, ax=axes, shrink=.7, label="korelasi Spearman")
save(fig, "09_korelasi")

# 10: pengaruh frame hilang, dibandingkan di dalam video yang sama
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, (field, title) in zip(axes, [("arus", "Arus terhitung"), ("di_roi", "Kendaraan terlihat di ROI")]):
    for i, key in enumerate(DIRS):
        vals = [S[key]["dalam_video"][n][field] for n in LOW_BINS]
        pos = np.arange(len(LOW_BINS)) + (i - .5) * .38
        ax.bar(pos, vals, width=.34, color=COLOR[key], label=DIRS[key])
        for p_, v_ in zip(pos, vals):
            ax.text(p_, v_ + .02, f"{v_:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(LOW_BINS)), [f"frame ada {n}" for n in LOW_BINS])
    ax.axhline(1, color=INK, linewidth=.8, linestyle=(0, (3, 2)))
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Rasio terhadap menit utuh (video sama)")
axes[0].set_ylim(0, 1.4)
axes[0].legend(ncol=2, loc="upper left")
fig.tight_layout()
save(fig, "10_arus_vs_mutu")

# 11: kondisi lalu lintas per jam
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, (key, m) in zip(axes, frames.items()):
    g = m[m.layak].groupby("jam")[["pct_time_padat", "pct_time_macet"]].mean().reindex(range(24)).fillna(0)
    ax.bar(g.index, g.pct_time_padat, color="#fab219", width=.75, label="Padat", edgecolor=SURFACE, linewidth=.6)
    ax.bar(g.index, g.pct_time_macet, bottom=g.pct_time_padat, color="#d03b3b", width=.75, label="Macet",
           edgecolor=SURFACE, linewidth=.6)
    ax.set_title(DIRS[key])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Jam (WIB)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
axes[0].set_ylabel("% waktu (sisanya lancar)")
fig.tight_layout()
save(fig, "11_kondisi")

print(json.dumps(S, indent=1, default=str))
