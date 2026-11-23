# Laporan — Klasifikasi Cuaca CCTV (Hujan / Cerah) dengan ML & DL

**Cakupan:** kamera SP Buah Batu (640×480), klasifikasi biner **hujan (rain)** vs **cerah (sunny)**, luring, hanya CPU.
**Data:** `weather/dataset/sp_buahbatu_rain.mp4` (19,1 MB, 14.996 frame ≈ 600 dtk) dan `weather/dataset/sp_buahbatu_sunny.mp4` (186,7 KB, 125 frame, 5 dtk).
**Peralatan:** `script_tools/weather_ml.py` (`slice | split | train --model {rf,cnn} | evaluate`).
**Keluaran:** model klasik acuan + RandomForest + CNN dari nol (PyTorch), semuanya dievaluasi pada split uji yang sama.

---

## Langkah 1 — Slice: membuat dataset frame yang seimbang

**Perintah:** `python -m script_tools.weather_ml slice`
**Hasil:** `rain=120 sunny=120` → `weather/dataset_frames/all/{rain,sunny}/frame_*.jpg` (JPEG kualitas 90).

**Mengapa langkah ini diperlukan?** Model belajar dari pasangan gambar→label. Karena itu video harus lebih dulu diubah menjadi frame gambar, lalu dikelompokkan ke folder per kelas.

**Keputusan penting:**
- **Penyeimbangan struktural di sumber, bukan secara statistik belakangan.** Klip hujan berisi 14.996 frame, sedangkan klip cerah hanya 125 frame. Jika semua frame diambil, model akan melihat hujan sekitar 120× lebih sering daripada cerah sehingga bias. Oleh karena itu penyeimbangan dilakukan *saat slicing*: tepat **120 frame per kelas** (batas atas = 125 frame milik klip cerah). Hujan diambil dengan subsampling seragam sepanjang waktu, cerah diambil hampir seluruhnya.
- **Crop anti kebocoran informasi.** Setiap frame dipotong ke persegi bebas teks `x∈[0,02 ; 0,98]·w`, `y∈[0,08 ; 0,92]·h`. Alasannya: stempel waktu yang tercetak di video terbaca `19-09-2026` (hujan) vs `05-06-2026` (cerah). Tanpa crop, CNN akan menghafal *tanggal*, bukan *cuaca* → kebocoran data → metrik yang tampak sempurna secara keliru. Crop memaksa model fokus pada kondisi jalan.
- **Kontrol galat:** klip tidak terbaca atau nol frame ter-decode → `ERROR: cannot decode ...` dengan kode keluar 2.

**Visualisasi:** `weather/_ml_slice_grid.png` — grid 2×3 (baris hujan + baris cerah) dengan poligon ROI jalan berwarna biru.

---

## Langkah 2 — Split latih / uji (stratified, dapat direproduksi)

**Perintah:** `python -m script_tools.weather_ml split`
**Hasil:** `train: rain=96 sunny=96 | test: rain=24 sunny=24` (file **dipindahkan** ke `dataset_frames/{train,test}/{rain,sunny}/`).

**Mengapa langkah ini diperlukan?** Model harus dievaluasi pada gambar yang **belum pernah dilihat** saat pelatihan; kalau tidak, yang diukur adalah hafalan, bukan generalisasi.

**Keputusan penting:**
- **Split bertingkat per kelas** (80% / 20%) dengan `random_state=42` → rasio hujan/cerah sama di train dan test, dan pembagiannya dapat direproduksi.
- **Berbasis file (pemindahan), bukan indeks.** Setiap JPEG secara fisik dipindah ke folder train/test — ini menjamin train/validasi/test tetap terpisah secara ketat, apa pun model yang dilatih kemudian.
- Pengaman: sel kosong (satu kelas tidak ada di train atau test) → berhenti, bukannya melatih pada satu kelas.

**Visualisasi:** `weather/_ml_split_dist.png` — bar berkelompok berisi jumlah train (biru) / test (abu-abu) per kelas.

---

## Langkah 3 — Acuan klasik: RandomForest (fitur citra)

**Perintah:** `python -m script_tools.weather_ml train --model rf`
**Hasil:** `rf_weather.joblib` + `metrics_rf.json`.

**Fitur (penggunaan ulang jalur klasik):** untuk setiap frame, dihitung di wilayah jalan (ROI) pada resolusi yang diperkecil:
- `spec_ratio` — proporsi piksel “berkilau” (pantulan spekular aspal basah),
- `road_gray` — kecerahan rata-rata jalan,
- `sat_mean` — saturasi rata-rata (hujan menurunkan saturasi warna),
- `lap_var` — varians Laplacian (ketajaman tekstur).

**Model:** `RandomForestClassifier(n_estimators=300, random_state=42)` — ensemble 300 pohon keputusan.

**Metrik pada test (48 frame):**

| | Precision | Recall | F1 |
|---|---|---|---|
| Hujan | 1,000 | 0,958 | 0,979 |
| Cerah | 0,960 | 1,000 | 0,980 |
| **Macro** | **0,980** | **0,979** | **0,979** |

- **Akurasi:** **0,979** (47/48 benar).
- **Confusion:** TP=23, FN=1, FP=0, TN=24 — hanya satu kesalahan: satu frame hujan diklasifikasikan sebagai cerah.

**Mengapa berhasil?** RandomForest menggabungkan pandangan statistik sederhana (kecerahan, saturasi, kilau, tekstur) dan tangguh terhadap data kecil; keempat fitur tidak memerlukan normalisasi (pohon tidak peka terhadap skala).

**Visualisasi:** `_cm_rf.png` (heatmap confusion), `_metrics_rf.png` (bar precision/recall/F1 + badge akurasi), `_featimp_rf.png` (kepentingan fitur).

---

## Langkah 4 — Deep learning: CNN dari nol (PyTorch, CPU)

**Perintah:** `python -m script_tools.weather_ml train --model cnn`
**Hasil:** `cnn_weather.pt` (bobot), `cnn_norm.json` (normalisasi), `metrics_cnn.json`.

**Pra-pemrosesan:** gambar hasil crop diubah ke **64×64 RGB**, dinormalisasi per kanal dengan mean/std yang **dihitung hanya dari train** (test tidak bocor).

**Arsitektur (literal, tanpa pencarian hiperparameter):**
```
Conv(3→16, k3, pad1) → ReLU → MaxPool(2)
Conv(16→32, k3)      → ReLU → MaxPool(2)
Conv(32→64, k3)      → ReLU → AdaptiveAvgPool(1×1)
Dropout(0.25) → Linear(64 → 2)
```

**Pelatihan:**
- Augmentasi `RandomHorizontalFlip(p=0.5)` hanya pada data latih.
- Split validasi 85/15 secara stratified (~163 fit / ~29 val).
- `Adam(lr=1e-3, weight_decay=1e-4)`, `CrossEntropyLoss`, batch 16, **20 epoch**, `torch.manual_seed(42)`.
- `state_dict` dengan **val-loss terbaik** yang disimpan (anti-overfitting).
- Kurva: training loss turun 0,687 → 0,029, akurasi validasi mencapai **1,00** sejak epoch ke-8.

**Metrik pada test (48 frame):**

| | Precision | Recall | F1 |
|---|---|---|---|
| Hujan | 1,000 | 1,000 | 1,000 |
| Cerah | 1,000 | 1,000 | 1,000 |
| **Macro** | **1,000** | **1,000** | **1,000** |

- **Akurasi:** **1,000** (48/48) — confusion sepenuhnya diagonal.

**Visualisasi:** `_curves_cnn.png` (train/val loss + akurasi val per epoch), `_cm_cnn.png`, `_metrics_cnn.png`.

---

## Langkah 5 — Perbandingan akhir dan evaluasi

**Perintah:** `python -m script_tools.weather_ml evaluate` → `_compare_models.png` + tabel konsol.

| Model | Akurasi | Macro F1 | F1 hujan | F1 cerah |
|---|---|---|---|---|
| Klasik (acuan in-sample) | 0,867 | 0,865 | — | — |
| RandomForest | 0,979 | 0,979 | 0,979 | 0,980 |
| CNN | 1,000 | 1,000 | 1,000 | 1,000 |

**Interpretasi:**
- **Kedua** model ML/DL melampaui baseline klasik (0,867): RF +0,112, CNN +0,133 akurasi.
- CNN (mempelajari fitur sepenuhnya otomatis) mengalahkan RF (fitur buatan manual) — deep learning menangkap sinyal yang tidak sepenuhnya tertangkap oleh 4 fitur manual.

**Arti metrik:**
- **Akurasi** = proporsi prediksi benar secara keseluruhan.
- **Precision (per kelas)** = dari frame yang diprediksi sebagai “X”, berapa yang benar-benar X (menghukum false positive).
- **Recall (per kelas)** = dari frame yang benar-benar X, berapa yang berhasil ditemukan (menghukum false negative).
- **F1** = rata-rata harmonik precision/recall (kompromi yang baik untuk kelas seimbang, seperti di sini).
- **Macro F1** = rata-rata F1 kedua kelas, tidak dipengaruhi ukuran kelas.

---

## Keterbatasan (perlu diketahui)

1. **Nilai 1,000 pada CNN bersifat optimistis.** Frame uji “cerah” berasal dari sekuens 5 detik yang hampir identik (near-duplicate). Evaluasi level-frame menampilkan tingkat kesulitan yang lebih rendah dari sebenarnya; generalisasi **level-klip** ke rekaman baru **belum terbukti**.
2. **Data sangat kecil.** 192 gambar latih sangat sedikit untuk CNN; variansinya sangat besar. Jika tersedia klip tambahan per kelas lalu langkah 1–5 dijalankan ulang *tanpa redesain*, akan diperoleh angka yang andal.
3. **RF kokoh, CNN sensitif.** RF (0,979) bertumpu pada fitur yang stabil secara fisik; jika skor CNN turun pada data baru, RF adalah pilihan paling aman di antara keduanya.
4. **Dependensi dev.** `scikit-learn`/`torch` hanya dipasang di mesin dev; `requirements.txt` dan `Dockerfile` tidak diubah (image deploy CPU yang ramping tetap utuh). Lingkungan dapat direkonstruksi via `weather/models_ml/TRAINING_ENV.txt`.

---

## Reproduksi penuh (dari root repo)

```bash
pip install scikit-learn torch --index-url https://download.pytorch.org/whl/cpu
python -m script_tools.weather_ml slice
python -m script_tools.weather_ml split
python -m script_tools.weather_ml train --model rf
python -m script_tools.weather_ml train --model cnn
python -m script_tools.weather_ml evaluate
```