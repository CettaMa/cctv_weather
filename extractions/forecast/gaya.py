"""Gaya grafik bersama (sama dengan eda/analisis.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import GAMBAR

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ARAH_WARNA = {"menjauhi": "#2a78d6", "mendekati": "#eb6834"}
ARAH_NAMA = {"menjauhi": "Menjauhi kamera", "mendekati": "Mendekati kamera"}
# warna mengikuti model (urutan kategorikal tetap), baseline abu-abu dengan pola garis
WARNA = {
    "lightgbm": "#2a78d6", "xgboost": "#eb6834", "lstm": "#1baf7a", "nhits": "#eda100", "tft": "#e87ba4",
    "tide": "#008300", "patchtst": "#6250d6", "itransformer": "#e34948", "ensemble": INK,
    "profil_historis": MUTED, "seasonal_naive": MUTED, "persistence": "#c3c2b7",
}
GARIS = {"profil_historis": (0, (4, 2)), "seasonal_naive": (0, (1, 1.5)), "persistence": (0, (6, 2, 1, 2))}
NAMA = {
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "lstm": "LSTM", "nhits": "N-HiTS", "tft": "TFT", "tide": "TiDE",
    "patchtst": "PatchTST", "itransformer": "iTransformer", "ensemble": "Ensemble", "profil_historis": "Profil historis",
    "seasonal_naive": "Seasonal naive", "persistence": "Persistence",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.titlelocation": "left", "legend.frameon": False,
})


def keluarga(kolom: str) -> str:
    return kolom.split("_")[0] if kolom.split("_")[0] in WARNA else kolom


def simpan(fig, nama: str) -> None:
    GAMBAR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GAMBAR / f"{nama}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
