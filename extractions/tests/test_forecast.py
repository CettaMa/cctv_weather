"""Uji penyiapan data dan fitur forecasting. Jalankan dengan venv induk (butuh pandas, lightgbm):
    ..\\.venv\\Scripts\\python.exe -m pytest tests\\test_forecast.py
"""
import pytest

pytest.importorskip("pandas")      # venv ekstraksi tidak punya pandas: lewati
import numpy as np
import pandas as pd
from forecast.common import DERET, MAX_FILL, TRAIN_END, TZ, VAL_END, bagian
from forecast.siapkan_data import isi_celah


def _seri(values):
    t = pd.date_range("2026-11-16 08:00", periods=len(values), freq="5min", tz=TZ)
    return pd.Series(values, dtype=float), pd.Series(np.full(len(values), 100.0)), t


def test_isi_celah_hanya_celah_pendek_yang_diapit():
    v = [1, np.nan, np.nan, 4] + [np.nan] * (MAX_FILL + 1) + [9, np.nan]
    s, prof, _ = _seri(v)
    hasil, panjang, sejak = isi_celah(s, prof)
    assert hasil[1:3].tolist() == pytest.approx([2, 3])          # celah 2 diisi linear (profil konstan)
    assert hasil[4:4 + MAX_FILL + 1].isna().all()                  # celah 7 dibiarkan kosong
    assert np.isnan(hasil.iloc[-1])                                # ekor tanpa pengapit tidak diisi
    assert panjang[4] == MAX_FILL + 1 and sejak.iloc[-1] == 1


def test_isi_celah_tidak_mengubah_nilai_teramati():
    s, prof, _ = _seri([5, np.nan, 7, 8])
    hasil, _, _ = isi_celah(s, prof)
    assert hasil[[0, 2, 3]].tolist() == [5, 7, 8]


def test_runs_untuk_awal_hujan():
    from forecast.deteksi_hujan import runs
    b = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1], bool)
    assert runs(b) == [(0, 3, False), (3, 5, True), (5, 6, False), (6, 9, True)]


def test_label_hujan_video_valid():
    from forecast.common import F
    lab = pd.read_csv(F / "label_hujan_video.csv")
    assert set(lab["label"]) <= {"kering", "basah"}
    assert (lab["dari"] <= lab["sampai"]).all()
    v = pd.read_csv(F / "awal_hujan_terverifikasi.csv")
    assert pd.to_datetime(v["awal"]).notna().all()


EKSPOR = __import__("pathlib").Path(__file__).resolve().parents[2] / "outputs" / "buahbatu_forecast.json"


@pytest.mark.skipif(not EKSPOR.exists(), reason="jalankan forecast.ekspor_app dulu")
def test_ekspor_app_konsisten():
    import json
    d = json.loads(EKSPOR.read_text(encoding="utf-8"))
    n = d["seri"]["n"]
    assert all(len(v) == n for k, v in d["seri"].items() if isinstance(v, list))
    assert len(d["mae_flow_per_h"]) == d["h_max"] == 72
    waktu = [a["t"] for a in d["asal"]]
    assert waktu == sorted(waktu) and d["asal_default"] in waktu
    for a in d["asal"]:
        assert 0 <= a["i"] < n
        for arah in ("menjauhi", "mendekati"):
            assert len(a["pred"][arah]["flow"]) == len(a["pred"][arah]["smp"]) == 72
            # nilai h=0 di panel adalah aktual pada titik asal, yang harus teramati
            assert d["seri"][f"flow_{arah}"][a["i"]] is not None
    # ramalan out-of-sample: asal 20-22 Nov dari model s.d. 19 Nov, 23-26 Nov dari model s.d. 22 Nov
    assert all(("19 Nov" in a["model"]) == (a["t"] < "2026-11-23") for a in d["asal"])


def test_bagian_kronologis():
    t = pd.Series([TRAIN_END - pd.Timedelta("5min"), TRAIN_END, VAL_END])
    assert bagian(t).tolist() == ["latih", "validasi", "uji"]


@pytest.mark.skipif(not DERET.exists(), reason="jalankan forecast.siapkan_data dulu")
def test_deret_grid_reguler_dan_target_tidak_diimputasi():
    from forecast.common import load_deret
    d = load_deret()
    for _, a in d.groupby("arah"):
        assert a["waktu"].is_unique
        assert (a["waktu"].diff().dropna() == pd.Timedelta("5min")).all()
        assert a.loc[~a["teramati"], "flow"].isna().all()
        assert (a.loc[a["diimputasi"], "panjang_celah"] <= MAX_FILL).all()


@pytest.mark.skipif(not DERET.exists(), reason="jalankan forecast.siapkan_data dulu")
def test_fitur_tidak_memakai_masa_depan():
    """Mengubah data setelah titik asal tidak boleh mengubah fitur non-kalender/non-cuaca baris itu."""
    from forecast.common import load_cuaca_5menit, load_deret
    from forecast.fitur import FITUR_DASAR, tabel
    d, c = load_deret(), load_cuaca_5menit()
    a = tabel(d, c, TRAIN_END)
    asal = a.loc[a["waktu_asal"] >= pd.Timestamp("2026-11-17 09:00", tz=TZ), "waktu_asal"].min()
    d2 = d.copy()
    masa_depan = d2["waktu"] > asal
    for k in [k for k in d2.columns if k.endswith("_isi") or k in ("flow", "smp")]:
        d2.loc[masa_depan, k] = d2.loc[masa_depan, k] * 3 + 7
    b = tabel(d2, c, TRAIN_END)
    # profil dihitung dari data sebelum cutoff (di sini TRAIN_END, setelah asal), jadi dikecualikan;
    # pada tahap validasi/uji cutoff selalu <= asal
    kol = [k for k in FITUR_DASAR if k not in ("profil_asal", "profil_target", "anomali0")]
    ka = a[a["waktu_asal"] == asal].sort_values(["arah", "h"])
    kb = b[b["waktu_asal"] == asal].sort_values(["arah", "h"])
    assert len(ka) > 0
    pd.testing.assert_frame_equal(ka[kol].reset_index(drop=True), kb[kol].reset_index(drop=True))
    # target ikut berubah, fitur tidak: memastikan perubahan memang terjadi
    assert not np.allclose(ka["y"].to_numpy(), kb["y"].to_numpy())


@pytest.mark.skipif(not DERET.exists(), reason="jalankan forecast.siapkan_data dulu")
def test_split_tidak_tumpang_tindih():
    from forecast.common import load_cuaca_5menit, load_deret
    from forecast.fitur import tabel
    from forecast.model_gbdt import tahap
    tab = tabel(load_deret(), load_cuaca_5menit(), VAL_END)
    tr, ev = tahap(tab, VAL_END, VAL_END, pd.Timestamp.max.tz_localize(TZ))
    assert tr["waktu_target"].max() < ev["waktu_asal"].min()
