"""Ekspor ramalan traffic dan cuaca titik CCTV Terusan Buah Batu - Jembatan Tol 02 untuk aplikasi peta.

Jalankan dari folder `extractions` (setelah siapkan_data, model_gbdt, cuaca --sumber openmeteo):
    ..\\.venv\\Scripts\\python.exe -m forecast.ekspor_app

Mode replay: titik asal = setiap bin 5 menit teramati pada 20-26 Nov. Ramalan arus dan SMP untuk
+5 sampai +360 menit (setiap 5 menit) dari LightGBM tanpa fitur cuaca, out-of-sample:
- asal 20-22 Nov memakai model yang dilatih dengan target sebelum 20 Nov,
- asal 23-26 Nov memakai model yang dilatih dengan target sebelum 23 Nov.
Cuaca: Open-Meteo per jam di titik CCTV, diinterpolasi ke 5 menit (data ramalan yang diarsipkan).
Keluaran: ../outputs/buahbatu_forecast.json (dibaca app.py).
"""
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import ARAH, DATA, HASIL, LAT, LON, ROOT, TRAIN_END, TZ, VAL_END, load_cuaca_5menit, load_deret
from .cuaca import ke_5menit
from .fitur import SKENARIO, tabel

OUT = ROOT.parent / "outputs" / "buahbatu_forecast.json"
H = list(range(1, 73))
FITUR = SKENARIO["tanpa_cuaca"]
NAMA_KAMERA = "CCTV TERUSAN BUAH BATU-JEMBATAN TOL 02"


def kapasitas() -> dict:
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from pkji.config import PKJIParams
        p = PKJIParams()
        return {"kapasitas_smp_jam": p.road_capacity_smp_per_hour, "dj_padat": p.dj_padat_threshold,
                "dj_macet": p.dj_macet_threshold, "lajur": p.active_lanes_in_roi}
    except Exception:   # nilai bawaan PKJIParams
        return {"kapasitas_smp_jam": 3264.0, "dj_padat": 0.45, "dj_macet": 0.85, "lajur": 2}


def latih(tab: pd.DataFrame, cut: pd.Timestamp, params: dict, n: int) -> lgb.Booster:
    tr = tab[tab["y"].notna() & (tab["waktu_target"] < cut)]
    p = {**params, "verbosity": -1, "seed": 7, "num_threads": 8}
    return lgb.train(p, lgb.Dataset(tr[FITUR], tr["y"], categorical_feature=["arah_kode"]), n)


def main() -> None:
    deret = load_deret()
    cuaca_fitur = load_cuaca_5menit()          # hanya untuk kolom cuaca tabel (tidak dipakai model)
    tune = json.loads((HASIL / "tuning_gbdt.json").read_text(encoding="utf-8"))["lightgbm"]
    params, n_iter = tune["params"], tune["iter"]

    pred = {}            # (waktu_asal, arah, target) -> array 72
    mae_rows = []
    for cut, lo, hi, nama in [(TRAIN_END, TRAIN_END, VAL_END, "latih s.d. 19 Nov"),
                              (VAL_END, VAL_END, None, "latih s.d. 22 Nov")]:
        for target in ("flow", "smp"):
            tab = tabel(deret, cuaca_fitur, cut, target=target, horizons=H, hanya_teramati=False)
            b = latih(tab, cut, params, n_iter)
            ev = tab[(tab["waktu_asal"] >= lo) & ((tab["waktu_asal"] < hi) if hi is not None else True)].copy()
            ev["p"] = np.clip(b.predict(ev[FITUR]), 0, None)
            if target == "flow":
                ok = ev["y"].notna()
                mae_rows.append(ev.loc[ok].assign(e=(ev["p"] - ev["y"]).abs())[["h", "e"]])
            for (t0, arah), g in ev.groupby(["waktu_asal", "arah"]):
                arr = np.full(len(H), np.nan)
                arr[g["h"].to_numpy() - 1] = g["p"].to_numpy()
                pred[(t0, arah, target)] = (arr, nama)
        print(nama, "selesai", flush=True)

    mae = pd.concat(mae_rows).groupby("h")["e"].mean()
    # deret 5 menit: aktual (hanya bin teramati) + cuaca Open-Meteo
    asal_t = sorted({k[0] for k in pred})
    mulai = asal_t[0] - pd.Timedelta("2h")
    akhir = asal_t[-1] + pd.Timedelta(minutes=5 * H[-1])
    grid = pd.date_range(mulai, akhir, freq="5min", tz=TZ)
    seri = {"mulai": mulai.isoformat(), "n": len(grid)}
    for arah in ARAH:
        a = deret[deret["arah"] == arah].set_index("waktu")
        for k in ("flow", "smp"):
            v = a[k].reindex(grid)
            seri[f"{k}_{arah}"] = [None if np.isnan(x) else round(float(x)) for x in v]
    a = deret[deret["arah"] == ARAH[0]].set_index("waktu")
    ulang = (a["n_menit_ulang"] > 0) & ~a["teramati"]      # bin tanpa aktual karena rekamannya ulangan
    seri["ulang"] = [int(x) for x in ulang.reindex(grid, fill_value=False)]
    om = ke_5menit(pd.read_csv(DATA / "cuaca_openmeteo.csv"), mulai, akhir).set_index("waktu")
    for k, d in (("hujan", 2), ("suhu", 1), ("rh", 0), ("awan", 0), ("angin", 1)):
        seri[k] = [None if pd.isna(x) else round(float(x), d) for x in om[k].reindex(grid)]

    idx = {t: i for i, t in enumerate(grid)}
    asal = []
    for t0 in asal_t:
        item = {"t": t0.isoformat(), "i": idx[t0], "model": pred[(t0, ARAH[0], "flow")][1], "pred": {}}
        for arah in ARAH:
            item["pred"][arah] = {k: [None if np.isnan(x) else round(float(x)) for x in pred[(t0, arah, k)][0]]
                                  for k in ("flow", "smp")}
        asal.append(item)

    # titik asal bawaan: yang galatnya paling tipikal (median) di antara asal dengan aktual cukup banyak,
    # supaya tampilan awal tidak memperlihatkan kasus terbaik maupun terburuk
    def galat(a):
        e = [abs(p - y) for arah in ARAH
             for p, y in zip(a["pred"][arah]["flow"], seri[f"flow_{arah}"][a["i"] + 1:])
             if p is not None and y is not None]
        return (len(e), float(np.mean(e)) if e else np.nan)
    nilai = [(galat(a), a["t"]) for a in asal]
    batas = np.quantile([n for (n, _), _ in nilai], 0.75)
    cukup = sorted((m, t) for (n, m), t in nilai if n >= batas)
    asal_default = cukup[len(cukup) // 2][1]

    out = {
        "kamera": {"nama": NAMA_KAMERA, "lat": LAT, "lon": LON, "arah": {"menjauhi": "Menjauhi kamera (arah Bojongsoang)",
                                                                         "mendekati": "Mendekati kamera"}},
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "langkah_menit": 5, "h_max": len(H),
        "model": "LightGBM direct multi-horizon tanpa fitur cuaca (forecast/ekspor_app.py)",
        "cuaca_sumber": "Open-Meteo historical-forecast per jam, diinterpolasi ke 5 menit",
        "mae_flow_per_h": [round(float(mae.get(h, np.nan)), 1) for h in H],
        **kapasitas(),
        "catatan": ("Mode replay 20-26 Nov 2026. Ramalan out-of-sample. Jumlah lajur dan kapasitas masih nilai "
                    "bawaan, jadi kondisi (DJ) hanya perkiraan. Aktual berasal dari deteksi video: saat streaming rusak "
                    "(lebih sering sejak 20 Nov) atau hujan lebat, aktual cenderung lebih rendah dari kenyataan."),
        "asal_default": asal_default, "seri": seri, "asal": asal,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} MB, {len(asal)} titik asal; MAE +5={mae.get(1):.0f}, "
          f"+60={mae.get(12):.0f}, +360={mae.get(72):.0f}")


if __name__ == "__main__":
    main()
