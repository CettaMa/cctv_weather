"""Ambil cuaca per jam lalu ubah ke grid 5 menit.

Jalankan dari folder `extractions`:
    ..\\.venv\\Scripts\\python.exe -m forecast.cuaca                     (OpenWeatherMap, default)
    ..\\.venv\\Scripts\\python.exe -m forecast.cuaca --sumber openmeteo  (Open-Meteo, tanpa API key)

OpenWeatherMap One Call 4.0 timeline (10 jam per panggilan), cadangan 3.0 timemachine (1 jam per
panggilan), sama dengan `thesis/forecasting/fc/weather_audit.py`. API key dibaca dari
$env:OPENWEATHER_API_KEY, atau dari file .env di `--env` (default thesis/forecasting/.env).
Hanya koordinat titik CCTV, waktu, dan API key yang dikirim. Data per jam disimpan di
forecast/data/cuaca_<sumber>.csv dan hanya jam yang belum ada yang diminta.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA, LAT, LON, TZ

AWAL, AKHIR = "2026-11-09", "2026-11-26"
WIB = timezone(timedelta(hours=7))
# folder `thesis` sejajar dengan folder proyek
ENV_DEFAULT = Path(__file__).resolve().parents[3] / "thesis" / "forecasting" / ".env"

OM_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_VARS = {"precipitation": "hujan", "temperature_2m": "suhu", "relative_humidity_2m": "rh",
           "cloud_cover": "awan", "wind_speed_10m": "angin", "weather_code": "kode_cuaca"}
OWM_40 = "https://api.openweathermap.org/data/4.0/onecall/timeline/1h"
OWM_30 = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
KOLOM = ["waktu", "hujan", "suhu", "rh", "awan", "angin", "kode_cuaca"]


def api_key(env: Path) -> str:
    key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "OPENWEATHER_API_KEY":
                key = v.strip().strip('"').strip("'")
    return key


def _get(url: str, params: dict) -> tuple[int, dict | str]:
    q = urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=30) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(1 + attempt)
                continue
            # jangan tampilkan URL (berisi API key)
            return e.code, e.read().decode("utf-8", "replace")[:200]
    return 0, "gagal"


def _baris_owm(item: dict) -> dict:
    rain = item.get("rain")
    hujan = float(rain.get("1h", 0) or 0) if isinstance(rain, dict) else float(rain or 0)
    w = (item.get("weather") or [{}])[0]
    angin = item.get("wind_speed")
    return {"waktu": datetime.fromtimestamp(int(item["dt"]), tz=timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:00"),
            "hujan": hujan, "suhu": item.get("temp"), "rh": item.get("humidity"), "awan": item.get("clouds"),
            "angin": None if angin is None else round(float(angin) * 3.6, 3), "kode_cuaca": w.get("id")}


def unduh_owm(path: Path, key: str, versi: str) -> dict:
    lama = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=KOLOM)
    ada = set(lama["waktu"].astype(str))
    jam = pd.date_range(f"{AWAL} 00:00", f"{AKHIR} 23:00", freq="h", tz=WIB)
    kurang = [t for t in jam if t.strftime("%Y-%m-%d %H:00") not in ada]
    stat = {"jam_target": len(jam), "sudah_ada": len(jam) - len(kurang), "panggilan": 0, "versi": versi}
    baru = []
    while kurang and key:
        t = kurang[0]
        if versi == "4.0":
            kode, js = _get(OWM_40, {"lat": LAT, "lon": LON, "start": int(t.timestamp()), "units": "metric", "appid": key})
        else:
            kode, js = _get(OWM_30, {"lat": LAT, "lon": LON, "dt": int(t.timestamp()), "units": "metric", "appid": key})
        if kode != 200:
            stat["galat"] = f"HTTP {kode}: {js}"
            break
        stat["panggilan"] += 1
        rows = [_baris_owm(it) for it in js.get("data", [])]
        rows = [r for r in rows if r["waktu"] not in ada and r["waktu"] <= f"{AKHIR} 23:00"]
        for r in rows:
            ada.add(r["waktu"])
        baru += rows
        sisa = [x for x in kurang if x.strftime("%Y-%m-%d %H:00") not in ada]
        kurang = [x for x in sisa if x > t] if (sisa and sisa[0] == t) else sisa   # jam yang tidak dikembalikan: lewati
        time.sleep(0.25)
    if baru:
        out = pd.concat([lama, pd.DataFrame(baru)], ignore_index=True).drop_duplicates("waktu", keep="last")
        out.sort_values("waktu").to_csv(path, index=False)
    stat["jam_baru"] = len(baru)
    return stat


def unduh_openmeteo(path: Path) -> dict:
    if path.exists():
        return {"jam_baru": 0}
    kode, js = _get(OM_URL, {"latitude": LAT, "longitude": LON, "start_date": AWAL, "end_date": AKHIR,
                             "hourly": ",".join(OM_VARS), "timezone": TZ})
    if kode != 200:
        raise RuntimeError(f"Open-Meteo HTTP {kode}: {js}")
    h = js["hourly"]
    d = pd.DataFrame({OM_VARS[k]: h[k] for k in OM_VARS})
    d.insert(0, "waktu", pd.to_datetime(h["time"]).strftime("%Y-%m-%d %H:00"))
    d.to_csv(path, index=False)
    return {"jam_baru": len(d)}


def ke_5menit(raw: pd.DataFrame, awal: pd.Timestamp, akhir: pd.Timestamp) -> pd.DataFrame:
    raw = raw.copy()
    raw["waktu"] = pd.to_datetime(raw["waktu"]).dt.tz_localize(TZ)
    raw = raw.set_index("waktu").sort_index()
    # jam yang tidak dikembalikan API (OWM: 19 Nov 00-05) diisi interpolasi antar-jam
    penuh = pd.date_range(raw.index.min(), raw.index.max(), freq="h", tz=TZ)
    raw = raw.reindex(penuh)
    num = ["hujan", "suhu", "rh", "awan", "angin"]
    raw[num] = raw[num].interpolate("time", limit_area="inside")
    raw["kode_cuaca"] = raw["kode_cuaca"].ffill()
    grid = pd.date_range(awal, akhir, freq="5min", tz=TZ)
    out = pd.DataFrame(index=grid)
    # hujan per jam = jumlah satu jam SEBELUM stempel waktu (Open-Meteo; OWM rain.1h juga "1 jam terakhir")
    # -> laju mm/jam berlaku (t-1j, t]; bin 5 menit [t, t+5m) masuk jam yang berakhir di ceil(t+5m)
    out["hujan"] = raw["hujan"].reindex((grid + pd.Timedelta("5min")).ceil("h")).to_numpy()
    for k in ["suhu", "rh", "awan", "angin"]:
        out[k] = raw[k].reindex(raw.index.union(grid)).interpolate("time").reindex(grid).to_numpy()
    out["kode_cuaca"] = raw["kode_cuaca"].reindex(grid, method="nearest").to_numpy()
    out["hujan_ya"] = (out["hujan"] >= 0.1).astype(int)
    out["hujan_1j"] = out["hujan"].rolling(12, min_periods=1).mean()          # mm dalam 1 jam terakhir
    out["hujan_3j"] = out["hujan"].rolling(36, min_periods=1).mean() * 3      # mm dalam 3 jam terakhir
    out["intensitas"] = pd.cut(out["hujan"], [-0.01, 0.1, 2.5, 7.6, np.inf],
                               labels=["kering", "ringan", "sedang", "lebat"]).astype(str)
    return out.rename_axis("waktu").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", choices=["owm", "openmeteo"], default="owm")
    ap.add_argument("--owm-versi", choices=["4.0", "3.0"], default="4.0")
    ap.add_argument("--env", type=Path, default=ENV_DEFAULT, help="file .env berisi OPENWEATHER_API_KEY")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"cuaca_{args.sumber}.csv"
    if args.sumber == "owm":
        key = api_key(args.env)
        if not key:
            raise SystemExit("OPENWEATHER_API_KEY tidak ditemukan (env atau --env).")
        stat = unduh_owm(path, key, args.owm_versi)
        if "galat" in stat and args.owm_versi == "4.0" and stat["panggilan"] == 0:
            print("One Call 4.0 gagal:", stat["galat"], "-> mencoba 3.0 timemachine")
            stat = unduh_owm(path, key, "3.0")
    else:
        stat = unduh_openmeteo(path)
    print("unduh:", stat)
    raw = pd.read_csv(path)
    jam = pd.date_range(f"{AWAL} 00:00", f"{AKHIR} 23:00", freq="h").strftime("%Y-%m-%d %H:00")
    hilang = int((~jam.isin(raw["waktu"])).sum())
    print(f"cuaca per jam: {len(raw)} baris, jam hilang: {hilang}, nilai kosong: {int(raw[KOLOM[1:6]].isna().sum().sum())}")
    if hilang:
        print("jam hilang (diisi interpolasi):", list(jam[~jam.isin(raw["waktu"])]))
    if hilang > 0.02 * len(jam):
        raise SystemExit("Terlalu banyak jam cuaca hilang; jalankan ulang untuk melanjutkan unduhan.")
    c = ke_5menit(raw, pd.Timestamp(f"{AWAL} 00:00", tz=TZ), pd.Timestamp(f"{AKHIR} 23:55", tz=TZ))
    c.to_csv(DATA / "cuaca_5menit.csv", index=False)
    info = {"sumber": args.sumber, "lat": LAT, "lon": LON, "jam": len(raw), "jam_hilang": hilang, **stat}
    info.pop("galat", None)
    # pembanding: korelasi hujan per jam dengan sumber lain bila filenya ada
    lain = DATA / f"cuaca_{'openmeteo' if args.sumber == 'owm' else 'owm'}.csv"
    if lain.exists():
        b = pd.read_csv(lain)
        m = raw.merge(b, on="waktu", suffixes=("", "_lain"))
        info["pembanding"] = {
            "sumber": lain.stem.replace("cuaca_", ""), "jam": len(m),
            **{f"korelasi_{k}": round(float(m[k].corr(m[f"{k}_lain"])), 3) for k in ["hujan", "suhu", "rh"]},
            "jam_hujan_ini": int((m["hujan"] >= 0.5).sum()), "jam_hujan_lain": int((m["hujan_lain"] >= 0.5).sum()),
            "total_hujan_ini_mm": round(float(m["hujan"].sum()), 1), "total_hujan_lain_mm": round(float(m["hujan_lain"].sum()), 1),
        }
    (DATA / "cuaca_sumber.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    print(c.describe().round(2).to_string())


if __name__ == "__main__":
    main()
