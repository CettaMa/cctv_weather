"""Uji deteksi rekaman berulang dari jam overlay (eda/pindai_ulang.py)."""
import pytest

pytest.importorskip("cv2")
import numpy as np

from eda.pindai_ulang import tandai_ulang


def test_putaran_ulang_ditandai():
    t = np.arange(0, 300, 10, dtype=float)
    # 100 detik pertama maju normal, lalu 100 detik yang sama diputar dua kali lagi
    jam = 1000 + (t % 100)
    status = tandai_ulang(t, jam)
    assert (status[:10] == 0).all()
    assert (status[11:20] == 1).all() and (status[21:] == 1).all()


def test_rekaman_normal_dan_salah_baca_tidak_ditandai():
    t = np.arange(0, 200, 10, dtype=float)
    jam = 5000 + t
    jam[5] = 100.0          # satu salah baca tidak membentuk runtun
    jam[8] = np.nan
    status = tandai_ulang(t, jam)
    assert status[5] == -1 and status[8] == -1
    assert (status[[0, 4, 6, 12, 19]] == 0).all()
